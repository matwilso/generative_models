import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from gms import utils

class VAE(utils.GM):
  DC = utils.AttrDict()  # default C
  DC.z_size = 128
  DC.beta = 1.0

  def __init__(self, C):
    super().__init__()
    self.C = C
    self.encoder = Encoder(C.z_size, C)
    self.decoder = Decoder(C.z_size, C)
    self.optimizer = Adam(self.parameters(), lr=C.lr)

  def loss(self, x):
    """VAE loss"""
    z_post = self.encoder(x)  # posterior  p(z|x)
    decoded = self.decoder(z_post.rsample())  # reconstruction p(x|z)
    recon_loss = -tdib.Bernoulli(logits=decoded).log_prob(x).mean((1, 2, 3))
    # kl div constraint
    z_prior = tdib.Normal(0, 1)
    kl_loss = tdib.kl_divergence(z_post, z_prior).mean(-1)
    # full loss and metrics
    loss = (recon_loss + self.C.beta * kl_loss).mean()
    metrics = {'loss': loss, 'recon_loss': recon_loss.mean(), 'kl_loss': kl_loss.mean()}
    return loss, metrics

  def sample(self, n):
    z = torch.randn(n, self.C.z_size).to(self.C.device)
    return self._decode(z)

  def evaluate(self, writer, x, epoch):
    """run samples and other evaluations"""
    samples = self.sample(25)
    writer.add_image('vae/samples', utils.combine_imgs(samples, 5, 5)[None], epoch)
    z_post = self.encoder(x[:8])
    recon = self._decode(z_post.mean)
    recon = torch.cat([x[:8].cpu(), recon], 0)
    writer.add_image('vae/reconstruction', utils.combine_imgs(recon, 2, 8)[None], epoch)
    writer.flush()

  def _decode(self, x):
    return 1.0 * (self.decoder(x).exp() > 0.5).cpu()

class Encoder(nn.Module):
  def __init__(self, out_size, C):
    super().__init__()
    self.C = C
    H = self.C.hidden_size
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
  def __init__(self, in_size, C):
    super().__init__()
    self.C = C
    H = self.C.hidden_size
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
