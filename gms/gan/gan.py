import sys

from matplotlib.pyplot import disconnect
import numpy as np
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from gms import utils
from gms import nets

class Discriminator(nn.Module):
  def __init__(self, out_size, C):
    super().__init__()
    self.C = C
    H = self.C.hidden_size
    self.net = nn.Sequential(
        nn.Conv2d(1, H, 3, 2),
        nn.LeakyReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(H, H, 3, 2),
        nn.LeakyReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(H, H, 3, 1),
        nn.LeakyReLU(),
        nn.Conv2d(H, out_size, 3, 2),
    )
  def forward(self, x):
    return self.net(x)

class Generator(nn.Module):
  def __init__(self, in_size, C):
    super().__init__()
    self.C = C
    H = self.C.hidden_size
    self.net = nn.Sequential(
        nn.ConvTranspose2d(self.C.z_size, H, 5, 1),
        nn.GroupNorm(32, H),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(H, H, 4, 2),
        nn.GroupNorm(32, H),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(H, H, 4, 2),
        nn.GroupNorm(32, H),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(H, 1, 3, 1),
        nn.Sigmoid(),
    )
  def forward(self, x):
    x = self.net(x[..., None, None])
    return x

# TODO: try out spec norm
class GAN(nn.Module):
  def __init__(self, C):
    super().__init__()
    self.disc = Discriminator(1, C)
    self.gen = Generator(C.z_size, C)
    # Setup Adam optimizers for both G and D
    self.disc_optim = Adam(self.disc.parameters(), lr=C.lr, betas=(0.5, 0.999))
    self.gen_optim = Adam(self.gen.parameters(), lr=C.lr, betas=(0.5, 0.999))
    self.C = C

  def train_step(self, batch):
    x = batch[0]
    # Train with all-real batch
    # Discriminator
    # real example
    self.disc_optim.zero_grad()
    real_disc = self.disc(x)
    # fake example
    noise = torch.randn(x.shape[0], self.C.z_size).to(self.C.device)
    fake = self.gen(noise)
    fake_disc = self.disc(fake)
    # backward
    disc_loss = (real_disc - fake_disc).mean()
    disc_loss += self.C.gan_reg * (real_disc**2 + fake_disc**2).mean()
    disc_loss.backward()
    #disc_grad_norm = nn.utils.clip_grad_norm_(self.disc.parameters(), 100)
    self.disc_optim.step()

    # Generator
    self.gen_optim.zero_grad()
    fake = self.gen(noise)
    fake_disc = self.disc(fake)
    gen_loss = fake_disc.mean()
    gen_loss.backward()
    #gen_grad_norm = nn.utils.clip_grad_norm_(self.gen.parameters(), 100)
    self.gen_optim.step()
    metrics = {'loss': disc_loss + gen_loss, 'disc_loss': disc_loss, 'gen_loss': gen_loss}#, 'disc_grad_norm': disc_grad_norm, 'gen_grad_norm': gen_grad_norm}
    return metrics

  def loss(self, *args, **kwargs):
    return torch.zeros(1), {}

  def sample(self, n):
    with torch.no_grad():
      noise = torch.randn(n, self.C.z_size).to(self.C.device)
      fake = self.gen(noise)
    return fake

  def evaluate(self, writer, batch, epoch):
    samples = self.sample(10)
    utils.plot_samples('samples', writer, epoch, batch[0][:10], samples)
    writer.flush()
