import sys

from matplotlib.pyplot import disconnect
import numpy as np
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from gms import utils
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
  def __init__(self, C):
    super().__init__()
    self.C = C
    H = self.C.hidden_size
    self.net = nn.Sequential(
        nn.Conv2d(1, H, 3, 2),
        nn.LeakyReLU(),
        nn.Conv2d(H, H, 3, 2),
        nn.BatchNorm2d(H),
        nn.LeakyReLU(),
        nn.Conv2d(H, H, 3, 1),
        nn.BatchNorm2d(H),
        nn.LeakyReLU(),
        nn.Conv2d(H, 1, 3, 2),
        nn.Sigmoid()
    )
    self.apply(weights_init)

  def forward(self, x):
    return self.net(x)

class Generator(nn.Module):
  def __init__(self, C):
    super().__init__()
    self.C = C
    H = self.C.hidden_size
    self.net = nn.Sequential(
        nn.ConvTranspose2d(self.C.noise_size, H, 5, 1),
        nn.BatchNorm2d(H),
        nn.ReLU(),
        nn.ConvTranspose2d(H, H, 4, 2),
        nn.BatchNorm2d(H),
        nn.ReLU(),
        nn.ConvTranspose2d(H, H, 4, 2),
        nn.BatchNorm2d(H),
        nn.ReLU(),
        nn.ConvTranspose2d(H, 1, 3, 1),
        nn.Sigmoid(),)
    self.apply(weights_init)

  def forward(self, x):
    x = self.net(x)
    return x

# TODO: try out spec norm
class GAN(nn.Module):
  DC = utils.AttrDict()  # default C
  DC.noise_size = 128
  DC.binarize = 0 # don't binarize the data for GAN, because then it's hard to backprop the gradients

  def __init__(self, C):
    super().__init__()
    self.disc = Discriminator(C)
    self.gen = Generator(C)
    # Setup Adam optimizers for both G and D
    self.disc_optim = Adam(self.disc.parameters(), lr=C.lr, betas=(0.5, 0.999))
    self.gen_optim = Adam(self.gen.parameters(), lr=C.lr, betas=(0.5, 0.999))
    self.fixed_noise = None
    self.real_label = 1
    self.fake_label = 0
    self.C = C
    self.criterion = nn.BCELoss()

  def train_step(self, batch):
    x = batch[0]
    bs = x.shape[0]
    # train with real
    self.disc_optim.zero_grad()
    label = torch.full((bs,1,1,1), self.real_label, dtype=x.dtype, device=x.device)
    output = self.disc(x)
    errD_real = self.criterion(output, label)
    errD_real.backward()
    D_x = output.mean().item()
    # train with fake
    noise = torch.randn(bs, self.C.noise_size, 1, 1, device=x.device)
    fake = self.gen(noise)
    label.fill_(self.fake_label)
    output = self.disc(fake.detach())
    errD_fake = self.criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    self.disc_optim.step()
    # gen
    self.gen_optim.zero_grad()
    label.fill_(self.real_label)  # fake labels are real for generator cost
    output = self.disc(fake)
    errG = self.criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()
    self.gen_optim.step()
    metrics = {}
    return metrics

  def loss(self, *args, **kwargs):
    return torch.zeros(1), {}

  def sample(self, n):
    if self.fixed_noise is None:
      self.fixed_noise = torch.randn(n, self.C.noise_size, 1, 1, device=self.C.device)
    with torch.no_grad():
      noise = self.fixed_noise
      fake = self.gen(noise)
    return fake

  def evaluate(self, writer, batch, epoch):
    samples = self.sample(10)
    utils.plot_samples('samples', writer, epoch, batch[0][:10], samples)
    writer.flush()