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
from gms.nets import E1, D1
from gms import utils

H = utils.AttrDict()
H.bs = 32
H.z_size = 128
H.bn = 0
H.device = 'cuda'
H.log_n = 1000
H.done_n = 20
H.b = 0.1
H.logdir = './logs/'
H.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
H.lr = 1e-4
H.class_cond = 0
H.hidden_size = 512
H.append_loc = 1
H.overfit_batch = 0

# TODO: record bits/dim
# TODO: try interpolation

def append_location(x):
  """add xy coords to every pixel"""
  XY = torch.stack(torch.meshgrid(torch.linspace(0, 1, x.shape[-2]), torch.linspace(0, 1, x.shape[-1])), 0)
  return torch.cat([x, XY[None].repeat_interleave(x.shape[0], 0).to(x.device)], 1)

# Implementation pulled from https://github.com/ryujaehun/wavenet
# Type 'B' Conv
class DilatedCausalConv1d(nn.Module):
  """Dilated Causal Convolution for WaveNet"""

  def __init__(self, mask_type, in_channels, out_channels, dilation=1):
    super(DilatedCausalConv1d, self).__init__()
    self.conv = nn.Conv1d(in_channels, out_channels,
                          kernel_size=2, dilation=dilation, padding=0)
    self.dilation = dilation
    self.mask_type = mask_type
    assert mask_type in ['A', 'B']

  def forward(self, x):
    import ipdb; ipdb.set_trace()
    if self.mask_type == 'A':
      return self.conv(F.pad(x, [2, 0]))[:, :, :-1]
    else:
      return self.conv(F.pad(x, [self.dilation, 0]))

class ResidualBlock(nn.Module):
  def __init__(self, res_channels, dilation):
    super(ResidualBlock, self).__init__()

    self.dilated = DilatedCausalConv1d('B', res_channels, 2 * res_channels, dilation=dilation)
    self.conv_res = nn.Conv1d(res_channels, res_channels, 1)

  def forward(self, x):
    output = self.dilated(x)

    # PixelCNN gate
    o1, o2 = output.chunk(2, dim=1)
    output = torch.tanh(o1) * torch.sigmoid(o2)
    output = x + self.conv_res(output)  # Residual network

    return output

class Wavenet(nn.Module):
  def __init__(self, H):
    super().__init__()
    in_channels = 3
    out_channels = 1
    res_channels = 64
    layer_size = 9  # Largest dilation is 512
    stack_size = 1
    self.causal = DilatedCausalConv1d('A', in_channels, res_channels, dilation=1)
    self.res_stack = nn.Sequential(*[ResidualBlock(res_channels, 2 ** i) for i in range(layer_size)])
    self.out_conv = nn.Conv1d(res_channels, out_channels, 1)
    self.H = H

  def forward(self, x):
    batch_size = x.shape[0]
    x = append_location(x)
    output = x.view(batch_size, -1, 784)
    output = self.causal(output)
    output = self.res_stack(output)
    output = self.out_conv(output)
    return output.view(batch_size, 1, 28, 28)

  def nll(self, x):
    x = x[0]
    logits = self.forward(x)
    return F.binary_cross_entropy_with_logits(logits, x)

  def sample(self, n):
    with torch.no_grad():
      samples = torch.zeros(n, 1, 28, 28).to(self.H.device)
      for r in range(28):
        for c in range(28):
          logits = self(samples)[:, :, r, c]
          probs = torch.sigmoid(logits)
          samples[:, :, r, c] = torch.bernoulli(probs)
    return samples.cpu()


if __name__ == '__main__':
  # TODO: use low beta 0.1
  # TODO: make network bigger
  from gms.utils import load_mnist
  H = utils.parseH(H)
  writer = SummaryWriter(H.logdir)
  logger = utils.dump_logger({}, writer, 0, H)
  train_ds, test_ds = load_mnist(H.bs)
  _batch = next(iter(train_ds))
  _batch[0] = _batch[0].to(H.device)
  model = Wavenet(H).to(H.device)
  optimizer = Adam(model.parameters(), lr=H.lr)

  def train_epoch():
    if H.overfit_batch:
      for i in range(H.log_n):
        optimizer.zero_grad()
        loss = model.nll(_batch)
        loss.backward()
        optimizer.step()
        logger['loss'] += [loss.detach().cpu()]
    else:
      for batch in train_ds:
        batch[0], batch[1] = batch[0].to(H.device), batch[1].to(H.device)
        optimizer.zero_grad()
        loss = model.nll(batch)
        loss.backward()
        optimizer.step()
        logger['loss'] += [loss.detach().cpu()]

  def eval():
    model.eval()
    if H.overfit_batch:
      batch = _batch
      loss = model.nll(batch)
      logger['test/bits_per_dim'] = loss.item() / np.log(2)
    else:
      total_loss = 0
      with torch.no_grad():
        for batch in test_ds:
          batch[0], batch[1] = batch[0].to(H.device), batch[1].to(H.device)
          loss = model.nll(batch)
          total_loss += loss * batch[0].shape[0]
        avg_loss = total_loss / len(test_ds.dataset)
      logger['test/bits_per_dim'] = avg_loss.item() / np.log(2)
    samples = model.sample(10)
    utils.plot_samples(writer, i, batch[0][:10], samples)
    writer.flush()
    model.train()

  for i in count():
    train_epoch()
    eval()
    logger = utils.dump_logger(logger, writer, i, H)

    if i >= H.done_n:
      break
