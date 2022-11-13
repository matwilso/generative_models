import torch
import torch.nn.functional as F
from torch import distributions as tdib
from torch import nn
from torch.optim import Adam

from gms import common


class VAE(common.GM):
    DG = common.AttrDict()  # default G
    DG.z_size = 128
    DG.beta = 1.0

    def __init__(self, G):
        super().__init__(G)
        self.encoder = Encoder(G.z_size, G)
        self.decoder = Decoder(G.z_size, G)
        self.optimizer = Adam(self.parameters(), lr=G.lr)

    def loss(self, x, y=None):
        """VAE loss"""
        z_post = self.encoder(x)  # posterior  p(z|x)
        decoded = self.decoder(z_post.rsample())  # reconstruction p(x|z)
        if self.G.binarize:
            recon_loss = -tdib.Bernoulli(logits=decoded).log_prob(x).mean((1, 2, 3))
        else:
            recon_loss = -tdib.Normal(decoded, 1).log_prob(x).mean((1, 2, 3))
        # kl div constraint
        z_prior = tdib.Normal(0, 1)
        kl_loss = tdib.kl_divergence(z_post, z_prior).mean(-1)
        # full loss and metrics
        loss = (recon_loss + self.G.beta * kl_loss).mean()
        metrics = {
            'vae_loss': loss,
            'recon_loss': recon_loss.mean(),
            'kl_loss': kl_loss.mean(),
        }
        return loss, metrics

    def sample(self, n):
        z = torch.randn(n, self.G.z_size).to(self.G.device)
        return self._decode(z)

    def evaluate(self, writer, x, y, epoch):
        """run samples and other evaluations"""
        samples = self.sample(25)
        writer.add_image('samples', common.combine_imgs(samples, 5, 5)[None], epoch)
        z_post = self.encoder(x[:8])
        truth = x[:8].cpu()
        recon = self._decode(z_post.mean)
        error = (recon - truth + 1.0) / 2.0
        stack = torch.cat([truth, recon, error], 0)
        writer.add_image('reconstruction', common.combine_imgs(stack, 3, 8)[None], epoch)

    def _decode(self, x):
        return 1.0 * (torch.sigmoid(self.decoder(x)) > 0.5).cpu()


class Encoder(nn.Module):
    def __init__(self, out_size, G):
        super().__init__()
        H = G.hidden_size
        self.net = nn.Sequential(
            nn.Conv2d(1, H, 3, 2),
            nn.ReLU(),
            nn.Conv2d(H, H, 3, 2),
            nn.ReLU(),
            nn.Conv2d(H, H, 3, 1),
            nn.ReLU(),
            nn.Conv2d(H, 2 * out_size, 3, 2),
            nn.Flatten(1, 3),
        )

    def get_dist(self, x):
        mu, log_std = x.chunk(2, -1)
        std = F.softplus(log_std) + 1e-4
        return tdib.Normal(mu, std)

    def forward(self, x):
        return self.get_dist(self.net(x))


class Decoder(nn.Module):
    def __init__(self, in_size, G):
        super().__init__()
        H = G.hidden_size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_size, H, 5, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(H, H, 4, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(H, H, 4, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(H, 1, 3, 1),
        )

    def forward(self, x):
        x = self.net(x[..., None, None])
        return x
