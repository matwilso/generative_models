import numpy as np
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from gms import utils
from gms import nets

class VAE(nn.Module):
  def __init__(self, C):
    super().__init__()
    self.encoder = nn.Sequential(
        nets.DownConv(2 * C.z_size, C),
        nn.Flatten(1, 3),
        nets.GaussHead()
    )
    self.decoder = nets.UpConv(C.z_size, C)
    self.optimizer = Adam(self.parameters(), lr=C.lr)
    self.C = C

  def train_step(self, batch):
    self.optimizer.zero_grad()
    loss, metrics = self.loss(batch)
    loss.backward()
    self.optimizer.step()
    return metrics

  def sample(self, n):
    z = torch.randn(n, self.C.z_size).to(self.C.device)
    return 1.0 * self.decoder(z).exp() > 0.5

  def evaluate(self, writer, batch, epoch):
    samples = self.sample(10)
    utils.plot_samples('samples', writer, epoch, batch[0][:10], samples)
    z_post = self.encoder(batch[0][:10])
    reconmu = 1.0 * self.decoder(z_post.mean).exp() > 0.5
    utils.plot_samples('reconmu', writer, epoch, batch[0][:10], reconmu)
    writer.flush()

  def loss(self, x):
    x = x[0]
    # encode to compute approx posterior p(z|x)
    z_post = self.encoder(x)
    z_prior = tdib.Normal(torch.zeros_like(z_post.mean), torch.ones_like(z_post.stddev))
    kl_loss = tdib.kl_divergence(z_post, z_prior).mean(-1)
    # sample and decode the z to get the reconstruction p(x|z)
    decoded = self.decoder(z_post.rsample())
    recon_loss = -tdib.Bernoulli(logits=decoded).log_prob(x).mean((1, 2, 3))
    #recon_loss = -tdib.Normal(decoded, torch.ones_like(decoded)).log_prob(x).mean((1,2,3))
    # full loss and metrics
    loss = (recon_loss + self.C.beta * kl_loss).mean()
    metrics = {'loss': loss, 'recon_loss': recon_loss.mean(), 'kl_loss': kl_loss.mean()}
    return loss, metrics

