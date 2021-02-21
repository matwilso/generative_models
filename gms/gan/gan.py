import sys
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
from itertools import chain, count
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
from gms import utils

H = utils.AttrDict()
H.bs = 512
H.z_size = 128
H.bn = 0
H.device = 'cuda'
H.log_n = 1000
H.done_n = 1e7
H.b = 0.1
H.logdir = './logs/'
H.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
H.lr = 1e-4
H.class_cond = 0
H.reg = 1e-4

# TODO: try out spec norm

class Discriminator(nn.Module):
  def __init__(self, H):
    super().__init__()
    self.c1 = nn.Conv2d(3, 64, 3, stride=2)
    self.c2 = nn.Conv2d(64, 512, 3, stride=2)
    self.c3 = nn.Conv2d(512, 512, 3, stride=2)
    self.c4 = nn.Conv2d(512, 512, 3, stride=1)
    self.f1 = nn.Linear(512, 1)

  def forward(self, x):
    x = self.c1(x)
    x = F.relu(x)
    x = self.c2(x)
    x = F.relu(x)
    x = self.c3(x)
    x = F.relu(x)
    x = self.c4(x)
    x = F.relu(x)
    x = x.flatten(1, 3)
    x = self.f1(x)
    #x = torch.sigmoid(x)
    return x


class Generator(nn.Module):
  def __init__(self, H):
    super().__init__()
    self.d1 = nn.ConvTranspose2d(H.z_size, 512, 3, stride=2)
    self.d2 = nn.ConvTranspose2d(512, 512, 3, stride=2)
    self.d3 = nn.ConvTranspose2d(512, 64, 3, stride=2)
    self.d4 = nn.ConvTranspose2d(64, 3, 4, stride=2)

  def forward(self, x):
    x = self.d1(x[..., None, None])
    x = F.relu(x)
    x = self.d2(x)
    x = F.relu(x)
    x = self.d3(x)
    x = F.relu(x)
    x = self.d4(x)
    return x


if __name__ == '__main__':
  from gms.utils import CIFAR, MNIST
  H = utils.parseC(H)
  ds = CIFAR(H)
  disc = Discriminator(H).to(H.device)
  gen = Generator(H).to(H.device)
  writer = SummaryWriter(H.logdir)
  logger = utils.dump_logger({}, writer, 0, H)

  # Setup Adam optimizers for both G and D
  disc_optim = Adam(disc.parameters(), lr=H.lr, betas=(0.5, 0.999))
  gen_optim = Adam(gen.parameters(), lr=H.lr, betas=(0.5, 0.999))

  #fixed_noise = torch.randn(64, nz, 1, 1, device=device)
  real_label = 1.
  fake_label = 0.

  for i in count():
    # Train with all-real batch
    # Discriminator
    # real example
    disc_optim.zero_grad()
    batch = ds.sample_batch(H.bs)
    real_disc = disc(batch['image'])
    # fake example
    noise = torch.randn(H.bs, H.z_size).to(H.device)
    fake = gen(noise)
    fake_disc = disc(fake)
    # backward
    disc_loss = (real_disc - fake_disc).mean()
    disc_loss += H.reg * (real_disc**2 + fake_disc**2).mean()
    #disc_loss = real_disc.mean() - fake_disc.mean()
    disc_loss.backward()
    disc_optim.step()

    # Generator
    gen_optim.zero_grad()
    fake = gen(noise)
    fake_disc = disc(fake)
    gen_loss = fake_disc.mean()
    gen_loss.backward()
    gen_optim.step()
    logger['disc_loss'] += [disc_loss.detach().cpu()]
    logger['gen_loss'] += [gen_loss.mean().detach().cpu()]

    if i % 1000 == 0:
      pred = fake[:10]
      truth = batch['image'][:10]
      utils.plot_samples(writer, i, pred, truth)
      print(i, disc_loss.item())
      logger = utils.dump_logger(logger, writer, i, H)
