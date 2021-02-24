import itertools
import torch
from torch import distributions as tdib
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gms import utils
from gms.autoreg.transformer import Block

class CategoricalHead(nn.Module):
  """take logits and produce a bernoulli distribution independently on each element of the token"""
  def __init__(self, in_n, out_n, C):
    super().__init__()
    self.C = C
    self.layer = nn.Linear(in_n, out_n)

  def forward(self, x, past_o=None):
    x = self.layer(x)
    return tdib.Multinomial(logits=x)

class TransformerCNN(nn.Module):
  """  the full GPT language model, with a context size of block_size """
  def __init__(self, in_size, block_size, C):
    super().__init__()
    self.block_size = block_size
    self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, C.n_embed))
    self.embed = nn.Linear(in_size, C.n_embed, bias=False)
    self.blocks = nn.Sequential(*[Block(block_size, C) for _ in range(C.n_layer)])
    self.ln_f = nn.LayerNorm(C.n_embed)
    self.dist_head = CategoricalHead(C.n_embed, in_size, C)
    self.C = C

  def forward(self, x):
    BS, T, C = x.shape
    # SHIFT RIGHT (add a padding on the left) so you can't see yourself 
    x = torch.cat([torch.zeros(BS, 1, C).to(self.C.device), x[:, :-1]], dim=1)
    # forward the GPT model
    x = self.embed(x)
    x += self.pos_emb # each position maps to a (learnable) vector
    # add padding on left so that we can't see ourself.
    x = self.blocks(x)
    logits = self.ln_f(x)
    return self.dist_head(logits)

  def sample(self, n):
    with torch.no_grad():
      batch = [torch.zeros(n, self.block_size).to(self.C.device)]
      for i in range(self.block_size-1):
        import ipdb; ipdb.set_trace()
        bindist = self.forward(batch)
        batch[0][:, i + 1] = bindist.sample()[:, i]
    import ipdb; ipdb.set_trace()
    return batch[0]

class VQVAE(nn.Module):
  DC = utils.AttrDict()  # default C
  DC.z_size = 64
  DC.vqK = 128
  DC.beta = 0.25
  DC.n_layer = 2
  DC.n_head = 4
  DC.n_embed = 128

  def __init__(self, C):
    super().__init__()
    H = C.hidden_size
    # encode image into continuous latent space
    self.encoder = Encoder(C)
    self.pre_quant = nn.Conv2d(H, C.z_size, kernel_size=1, stride=1)
    # pass continuous latent vector through discretization bottleneck
    self.vector_quantization = VectorQuantizer(C.vqK, C.z_size, C.beta, C)
    # decode the discrete latent representation
    self.decoder = Decoder(C)
    self.transformerCNN = TransformerCNN(C.vqK, 7*7, C)

    self.optimizer = Adam(self.parameters(), lr=C.lr)
    self.prior_optimizer = Adam(self.transformerCNN.parameters(), lr=C.lr)
    self.C = C

  def train_step(self, batch):
    x = batch[0]
    for p in self.transformerCNN.parameters():
      p.requires_grad = False
    self.optimizer.zero_grad()
    embed_loss, decoded, perplexity, idxs = self.forward(x)
    recon_loss = -tdib.Bernoulli(logits=decoded).log_prob(x).mean()
    loss = recon_loss + embed_loss
    loss.backward()
    self.optimizer.step()
    for p in self.transformerCNN.parameters():
      p.requires_grad = True
    idxs.detach()
    self.prior_optimizer.zero_grad()
    one_hot = F.one_hot(idxs, self.C.vqK).float()
    one_hot = one_hot.flatten(1,2)
    dist = self.transformerCNN.forward(one_hot)
    prior_loss = -dist.log_prob(one_hot).mean()
    prior_loss.backward()
    self.prior_optimizer.step()
    return {'loss': loss, 'recon_loss': recon_loss, 'embed_loss': embed_loss, 'prior_loss': prior_loss}

  def forward(self, x, verbose=False):
    z_e = self.encoder(x)
    z_e = self.pre_quant(z_e)
    embed_loss, z_q, perplexity, _, idxs = self.vector_quantization(z_e)
    decoded = self.decoder(z_q)
    return embed_loss, decoded, perplexity, idxs

  def evaluate(self, writer, batch, epoch):
    _, decoded, _, _ = self.forward(batch[0][:10])
    reconmu = 1.0 * decoded.exp() > 0.5
    utils.plot_samples('reconmu', writer, epoch, batch[0][:10], reconmu)
    writer.flush()

  def loss(self, batch):
    return torch.zeros(1), {}

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
        nn.Conv2d(H, H, 3, 1, padding=1),
        nn.ReLU(),
    )

  def forward(self, x):
    out = self.net(x)
    return self.net(x)

class Decoder(nn.Module):
  def __init__(self, C):
    super().__init__()
    self.C = C
    H = self.C.hidden_size
    self.net = nn.Sequential(
        nn.ConvTranspose2d(C.z_size, H, 6, 3),
        nn.ReLU(),
        nn.ConvTranspose2d(H, H, 3, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(H, H, 3, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(H, 1, 1, 1),
    )

  def forward(self, x):
    x = self.net(x)
    return x

class VectorQuantizer(nn.Module):
  def __init__(self, K, D, beta, C):
    super().__init__()
    self.K = K
    self.D = D
    self.beta = beta
    self.embedding = nn.Embedding(self.K, self.D)
    self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)
    self.C = C

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
    return loss, z_q, perplexity, min_encodings, min_encoding_indices.view(z.shape[:-1])