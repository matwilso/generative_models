import sys
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
from itertools import chain, count
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F

# TODO: VQ VAE may be worth doing. but maybe as a separate repo.
from nets import E1, D1
import utils

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
H.lr = 3e-4
H.class_cond = 0

if __name__ == '__main__':
  # TODO: use low beta 0.1
  # TODO: make network bigger
  from utils import CIFAR, MNIST
  from net2 import ResNet18Enc, ResNet18Dec
  H = utils.parseC(H)
  writer = SummaryWriter(H.logdir)
  logger = utils.dump_logger({}, writer, 0, H)

  ds = CIFAR(H)
  #ds = MNIST(device)
  encoder = E1(H).to(H.device)
  decoder = D1(H).to(H.device)
  #encoder = ResNet18Enc(H=H).to(H.device)
  #decoder = ResNet18Dec(H=H).to(H.device)
  optimizer = Adam(chain(encoder.parameters(), decoder.parameters()), lr=H.lr)

  for i in count():
    optimizer.zero_grad()
    batch = ds.sample_batch(H.bs)
    prior_loss, code, mu = encoder(batch['image'])
    if H.class_cond:
      class_c = torch.nn.functional.one_hot(batch['label'], 10).float()
      recondist = decoder(code, class_c)
    else:
      recondist = decoder(code)

    recon_loss = -recondist.log_prob(batch['image'])
    loss = (H.b * prior_loss + recon_loss.mean((-1, -2, -3))).mean()
    loss.backward()
    optimizer.step()
    logger['total_loss'] += [loss.detach().cpu()]
    logger['prior_loss'] += [prior_loss.mean().detach().cpu()]
    logger['recon_loss'] += [recon_loss.mean().detach().cpu()]

    if i % H.log_n == 0:
      encoder.eval()
      decoder.eval()
      logger = utils.dump_logger(logger, writer, i, H)
      if H.class_cond:
        reconmu = decoder(mu[:10], class_c[:10]).mean
        reconsamp = decoder(torch.randn(mu[:10].shape).to(H.device), class_c[:10]).mean
      else:
        reconmu = decoder(mu[:10]).mean
        reconsamp = decoder(torch.randn(mu[:10].shape).to(H.device)).mean
      utils.plot_samples(writer, i, batch['image'][:10], reconmu, reconsamp)
      writer.flush()
      encoder.train()
      decoder.train()
    if i >= H.done_n:
      break
