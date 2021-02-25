import itertools

from matplotlib.pyplot import install_repl_displayhook
import torch
from torch import distributions as tdib
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gms import utils
from gms.autoregs.transformer import TransformerCNN

class VQVAE(nn.Module):
  DC = utils.AttrDict()  # default C
  DC.vqD = 64
  DC.vqK = 64
  DC.beta = 0.25
  DC.n_layer = 2
  DC.n_head = 8
  DC.n_embed = 256
  DC.phase = 0

  def __init__(self, C):
    super().__init__()
    H = C.hidden_size
    # encoder -> VQ -> decoder, learned in phase 0
    self.encoder = Encoder(C)
    self.vq = VectorQuantizer(C.vqK, C.vqD, C.beta, C)
    self.decoder = Decoder(C)
    # prior, learned in phase 1
    self.transformerCNN = TransformerCNN(C.vqK, 7*7, C)
    if C.phase == 0:
      self.optimizer = Adam(self.parameters(), lr=C.lr)
      for p in self.transformerCNN.parameters():
        p.requires_grad = False
    else:
      # load weights from phase 1
      path = C.logdir / 'model.pt'
      print("LOADED MODEL", path)
      self.load_state_dict(torch.load(path))
      self.optimizer = Adam(self.transformerCNN.parameters(), lr=C.lr)
      for p in self.parameters():
        p.requires_grad = False
      for p in self.transformerCNN.parameters():
        p.requires_grad = True
    self.C = C

  def train_step(self, x):
    embed_loss = recon_loss = prior_loss = loss = torch.zeros(1)
    self.optimizer.zero_grad()
    embed_loss, decoded, perplexity, idxs = self.forward(x)
    if self.C.phase == 0: # encoder-decoder training phase
      recon_loss = -tdib.Bernoulli(logits=decoded).log_prob(x).mean()
      loss = recon_loss + embed_loss
      loss.backward()
      metrics = {'loss': loss, 'recon_loss': recon_loss, 'embed_loss': embed_loss, 'perplexity': perplexity}
    else: # prior training phase
      idxs.detach()
      one_hot = F.one_hot(idxs, self.C.vqK).float()
      one_hot = one_hot.flatten(1,2)
      dist = self.transformerCNN.forward(one_hot)
      prior_loss = -dist.log_prob(one_hot).mean()
      prior_loss.backward()
      metrics = {'prior_loss': prior_loss}
    self.optimizer.step()
    return metrics

  def forward(self, x):
    z_e = self.encoder(x)
    embed_loss, z_q, perplexity, idxs = self.vq(z_e)
    decoded = self.decoder(z_q)
    return embed_loss, decoded, perplexity, idxs

  def sample(self, n):
    """requires a trained prior"""
    prior_idxs = self.transformerCNN.sample(n)
    prior_enc = self.vq.idx_to_encoding(prior_idxs)
    prior_enc = prior_enc.reshape([n, 7, 7, -1]).permute(0, 3, 1, 2)
    decoded = self.decoder(prior_enc)
    return 1.0*(decoded.exp() > 0.5).cpu()

  def evaluate(self, writer, x, epoch):
    if self.C.phase == 0:
      _, decoded, _, _ = self.forward(x[:8])
      recon = 1.0 * (decoded.exp() > 0.5).cpu()
      recon = torch.cat([x[:8].cpu(), recon], 0)
      writer.add_image('vqvae/reconstruction', utils.combine_imgs(recon, 2, 8)[None], epoch)
    else:
      samples = self.sample(25)
      writer.add_image('vqvae/samples', utils.combine_imgs(samples, 5, 5)[None], epoch)
    writer.flush()

class Encoder(nn.Module):
  def __init__(self, C):
    super().__init__()
    self.C = C
    H = self.C.hidden_size
    self.net = nn.Sequential(
        nn.Conv2d(1, H, 3, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(H, H, 3, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(H, H, 3, 1, padding=1),
        nn.ReLU(),
        nn.Conv2d(H, C.vqD, 3, 1, padding=1),
        nn.ReLU(),
    )
  def forward(self, x):
    return self.net(x)

class Decoder(nn.Module):
  def __init__(self, C):
    super().__init__()
    self.C = C
    H = self.C.hidden_size
    self.net = nn.Sequential(
        nn.ConvTranspose2d(C.vqD, H, 6, 3),
        nn.ReLU(),
        nn.ConvTranspose2d(H, H, 3, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(H, H, 3, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(H, 1, 1, 1),
    )
  def forward(self, x):
    return self.net(x)

class VectorQuantizer(nn.Module):
  """from: https://github.com/MishaLaskin/vqvae"""
  def __init__(self, K, D, beta, C):
    super().__init__()
    self.K = K
    self.D = D
    self.beta = beta
    self.embedding = nn.Embedding(self.K, self.D)
    self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)
    self.C = C

  def idx_to_encoding(self, one_hots):
    z_q = torch.matmul(one_hots, self.embedding.weight)
    return z_q

  def forward(self, z):
    # reshape z -> (batch, height, width, channel) and flatten
    z = z.permute(0, 2, 3, 1).contiguous()
    z_flattened = z.view(-1, self.D)
    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
    d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
    # find closest encodings
    min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
    min_encodings = torch.zeros(min_encoding_indices.shape[0], self.K).to(self.C.device)
    min_encodings.scatter_(1, min_encoding_indices, 1)
    # get quantized latent vectors
    z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
    # compute loss for embedding
    loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
    # preserve gradients
    z_q = z + (z_q - z).detach()
    # perplexity
    e_mean = torch.mean(min_encodings, dim=0)
    perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
    # reshape back to match original input shape
    z_q = z_q.permute(0, 3, 1, 2).contiguous()
    return loss, z_q, perplexity, min_encoding_indices.view(z.shape[:-1])