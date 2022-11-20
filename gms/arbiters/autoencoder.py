import torch
from torch import distributions as tdib
from torch import nn
from torch.optim import Adam

from gms import common


class Autoencoder(common.Arbiter):
    """
    Autoencoder model used to produce a latent space for computing metrics in (ie. just used for evaluation)
    """

    DG = common.AttrDict()  # default G
    DG.eval_heavy = False
    DG.z_size = 64
    DG.beta = 1e-6
    DG.binarize = 0

    def __init__(self, G):
        super().__init__(G)
        self.encoder = Encoder(G.z_size, G)
        self.decoder = Decoder(G.z_size, G)
        self.optimizer = Adam(self.parameters(), lr=G.lr)

    def forward(self, x):
        return self.encoder(x)

    def loss(self, x, y=None):
        z = self.encoder(x)
        decoded = self.decoder(z)
        if self.G.binarize:
            recon_loss = -tdib.Bernoulli(probs=decoded).log_prob(x).mean((1, 2, 3))
        else:
            recon_loss = -tdib.Normal(decoded, 1).log_prob(x).mean((1, 2, 3))
        # kl div constraint to normalize the latent a bit (not sure if neccessary)
        z_post = tdib.Normal(z, 1)
        z_prior = tdib.Normal(0, 1)
        kl_loss = tdib.kl_divergence(z_post, z_prior).mean(-1)
        # full loss and metrics
        loss = (recon_loss + self.G.beta * kl_loss).mean()
        metrics = {
            'full_loss': loss,
            'recon_loss': recon_loss.mean(),
            'kl_loss': kl_loss.mean(),
            'z_mean': z.mean(),
            'z_std': z.std(),
        }
        return loss, metrics

    def evaluate(self, writer, x, y, epoch):
        """run samples and other evaluations"""
        z = self.encoder(x[:8])
        recon = self._decode(z)
        truth = x[:8].cpu()
        error = (recon - truth + 1.0) / 2.0
        stack = torch.cat([truth, recon, error], 0)
        writer.add_image('reconstruction', common.combine_imgs(stack, 3, 8)[None], epoch)

    def _decode(self, x):
        if self.G.binarize:
            return 1.0 * (torch.sigmoid(self.decoder(x)) > 0.5).cpu()
        else:
            return self.decoder(x).cpu()


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
            nn.Conv2d(H, out_size, 3, 2),
            nn.Flatten(1, 3),
        )

    def forward(self, x):
        return self.net(x)


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
        if G.binarize:
            self.net.add_module('sigmoid', nn.Sigmoid())
        else:
            self.net.add_module('tanh', nn.Tanh())
        self.G = G

    def forward(self, x):
        x = self.net(x[..., None, None])
        return x
