import numpy as np
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from gms import utils

class VAE(nn.Module):
  def __init__(self, C):
    super().__init__()
    self.encoder = Encoder(C)
    self.decoder = Decoder(C)
    self.C = C

  def sample(self, n):
    z = torch.randn(n, self.C.z_size).to(self.C.device)
    return 1.0*self.decoder(z).exp() > 0.5

  def evaluate(self, writer, batch, epoch):
    samples = self.sample(10)
    utils.plot_samples('samples', writer, epoch, batch[0][:10], samples)
    mu, std = self.encoder(batch[0][:10])
    reconmu = 1.0*self.decoder(mu).exp() > 0.5
    utils.plot_samples('reconmu', writer, epoch, batch[0][:10], reconmu)
    writer.flush()

  def loss(self, x):
    x = x[0]
    # encode to compute approx posterior p(z|x)
    mu, std = self.encoder(x)
    z_post = tdib.Normal(mu, std)
    z_prior = tdib.Normal(torch.zeros_like(mu), torch.ones_like(std))
    kl_loss = tdib.kl_divergence(z_post, z_prior).mean(-1)
    # sample and decode the z to get the reconstruction p(x|z)
    decoded = self.decoder(z_post.rsample())
    recon_loss = -tdib.Bernoulli(logits=decoded).log_prob(x).mean((1,2,3))
    #recon_loss = -tdib.Normal(decoded, torch.ones_like(decoded)).log_prob(x).mean((1,2,3))
    # full loss and metrics
    loss = (recon_loss + self.C.beta*kl_loss).mean() 
    metrics = {'loss': loss, 'recon_loss': recon_loss.mean(), 'kl_loss': kl_loss.mean()}
    return loss, metrics

class Encoder(nn.Module):
  def __init__(self, C):
    super().__init__()
    self.C = C
    H = self.C.hidden_size
    self.net = nn.Sequential(
        nn.Conv2d(1, H, 4, 2),
        nn.ReLU(),
        nn.Conv2d(H, H, 4, 2),
        nn.ReLU(),
        nn.Conv2d(H, 2 * self.C.z_size, 4, 3),
    )
    #self.net = nn.Sequential(
    #    nn.Conv2d(1, H, 3, 2),
    #    nn.ReLU(),
    #    nn.Conv2d(H, H, 3, 2),
    #    nn.ReLU(),
    #    nn.Conv2d(H, H, 3, 1),
    #    nn.ReLU(),
    #    nn.Conv2d(H, 2 * self.C.z_size, 3, 2),
    #)

  def forward(self, x):
    x = self.net(x)
    x = x.flatten(1, 3)
    mu, log_std = x.split(self.C.z_size, dim=1)
    std = F.softplus(log_std)
    return mu, std

class Decoder(nn.Module):
  def __init__(self, C):
    super().__init__()
    self.C = C
    H = self.C.hidden_size
    self.net = nn.Sequential(
        nn.ConvTranspose2d(self.C.z_size, H, 6, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(H, H, 3, 2),
        nn.ReLU(),
        nn.ConvTranspose2d(H, 1, 4, 2),
    )
    #self.net = nn.Sequential(
    #    nn.ConvTranspose2d(self.C.z_size, H, 5, 1),
    #    nn.ReLU(),
    #    nn.ConvTranspose2d(H, H, 4, 2),
    #    nn.ReLU(),
    #    nn.ConvTranspose2d(H, H, 4, 2),
    #    nn.ReLU(),
    #    nn.ConvTranspose2d(H, 1, 3, 1),
    #)

  def forward(self, x):
    x = self.net(x[..., None, None])
    return x