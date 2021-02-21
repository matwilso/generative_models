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

# TODO: try out spec norm
class GAN(nn.Module):
  def __init__(self, C):
    super().__init__()
    self.disc = nets.DownConv(1, C)
    self.gen = nets.UpConv(C.z_size, C)
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
    noise = torch.randn(self.C.bs, self.C.z_size).to(self.C.device)
    fake = self.gen(noise)
    fake_disc = self.disc(fake)
    # backward
    disc_loss = (real_disc - fake_disc).mean()
    disc_loss += self.C.reg * (real_disc**2 + fake_disc**2).mean()
    disc_loss.backward()
    self.disc_optim.step()

    # Generator
    self.gen_optim.zero_grad()
    fake = self.gen(noise)
    fake_disc = self.disc(fake)
    gen_loss = fake_disc.mean()
    gen_loss.backward()
    self.gen_optim.step()
    metrics = {'loss': disc_loss+gen_loss, 'disc_loss': disc_loss, 'gen_loss': gen_loss}
    return metrics

  def evaluate(self):
    if i % 1000 == 0:
      pred = fake[:10]
      truth = batch['image'][:10]
      utils.plot_samples(writer, i, pred, truth)
      print(i, disc_loss.item())
      logger = utils.dump_logger(logger, writer, i, C)