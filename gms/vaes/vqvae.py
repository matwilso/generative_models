import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as tdib
from torch.optim import Adam

from gms import common
from gms.autoregs.transformer import TransformerCNN


class VQVAE(common.GM):
    DG = common.AttrDict()  # default G
    DG.vqD = 64
    DG.vqK = 64
    DG.beta = 0.25
    DG.n_layer = 2
    DG.n_head = 8
    DG.n_embed = 256
    DG.prior_lr = 1e-3

    def __init__(self, G):
        super().__init__(G)
        # encoder -> VQ -> decoder
        self.encoder = Encoder(G)
        self.vq = VectorQuantizer(G.vqK, G.vqD, G.beta, G)
        self.decoder = Decoder(G)
        # prior. this is usually learned after the other stuff has been trained, but we do it all in one swoop.
        self.transformerCNN = TransformerCNN(
            in_size=G.vqK, block_size=7 * 7, head='cat', G=G
        )
        self.optimizer = Adam(self.parameters(), lr=G.lr)
        self.prior_optimizer = Adam(
            self.transformerCNN.parameters(), lr=G.prior_lr, betas=(0.5, 0.999)
        )

    def train_step(self, x):
        # ENC-VQ-DEC
        self.zero_grad()
        embed_loss, decoded, perplexity, idxs = self.forward(x)
        recon_loss = -tdib.Bernoulli(logits=decoded).log_prob(x).mean()
        loss = recon_loss + embed_loss
        loss.backward()
        self.optimizer.step()
        # PRIOR
        self.zero_grad()
        code_idxs = F.one_hot(idxs.detach(), self.G.vqK).float().flatten(1, 2)
        dist = self.transformerCNN.forward(code_idxs)
        prior_loss = -dist.log_prob(code_idxs).mean()
        prior_loss.backward()
        self.prior_optimizer.step()
        return {
            'vq_vae_loss': loss,
            'recon_loss': recon_loss,
            'embed_loss': embed_loss,
            'perplexity': perplexity,
            'prior_loss': prior_loss,
        }

    def forward(self, x):
        z_e = self.encoder(x)
        embed_loss, z_q, perplexity, idxs = self.vq(z_e)
        decoded = self.decoder(z_q)
        return embed_loss, decoded, perplexity, idxs

    def sample(self, n):
        prior_idxs = self.transformerCNN.sample(n)[0]
        prior_enc = self.vq.idx_to_encoding(prior_idxs)
        prior_enc = prior_enc.reshape([n, 7, 7, -1]).permute(0, 3, 1, 2)
        decoded = self.decoder(prior_enc)
        return 1.0 * (torch.sigmoid(decoded) > 0.5).cpu()

    def evaluate(self, writer, x, epoch):
        _, decoded, _, _ = self.forward(x[:8])
        recon = 1.0 * (torch.sigmoid(decoded) > 0.5).cpu()
        recon = torch.cat([x[:8].cpu(), recon], 0)
        writer.add_image(
            'reconstruction', common.combine_imgs(recon, 2, 8)[None], epoch
        )
        samples = self.sample(25)
        writer.add_image('samples', common.combine_imgs(samples, 5, 5)[None], epoch)


class Encoder(nn.Module):
    def __init__(self, G):
        super().__init__()
        H = G.hidden_size
        self.net = nn.Sequential(
            nn.Conv2d(1, H, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(H, H, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(H, H, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(H, G.vqD, 3, 1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, G):
        super().__init__()
        H = G.hidden_size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(G.vqD, H, 6, 3),
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

    def __init__(self, K, D, beta, G):
        super().__init__()
        self.K = K
        self.D = D
        self.beta = beta
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)

    def idx_to_encoding(self, one_hots):
        z_q = torch.matmul(one_hots, self.embedding.weight)
        return z_q

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.D)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.K).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return loss, z_q, perplexity, min_encoding_indices.view(z.shape[:-1])
