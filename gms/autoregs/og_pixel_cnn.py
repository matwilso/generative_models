import torch.nn as nn
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
H.n_filters = 64
H.n_layers = 5
H.kernel_size = 7
H.use_resblock = 0


class MaskConv2d(nn.Conv2d):
  def __init__(self, mask_type, *args, **kwargs):
    assert mask_type == 'A' or mask_type == 'B'
    super().__init__(*args, **kwargs)
    self.register_buffer('mask', torch.zeros_like(self.weight))
    self.create_mask(mask_type)

  def forward(self, input, cond=None):
    batch_size = input.shape[0]
    out = F.conv2d(input, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
    return out

  def create_mask(self, mask_type):
    k = self.kernel_size[0]
    self.mask[:, :, :k // 2] = 1
    self.mask[:, :, k // 2, :k // 2] = 1
    if mask_type == 'B':
      self.mask[:, :, k // 2, k // 2] = 1

class ResBlock(nn.Module):
  def __init__(self, in_channels, **kwargs):
    super().__init__()
    self.block = nn.ModuleList([
        nn.ReLU(),
        MaskConv2d('B', in_channels, in_channels // 2, 1, **kwargs),
        nn.ReLU(),
        MaskConv2d('B', in_channels // 2, in_channels // 2, 7, padding=3, **kwargs),
        nn.ReLU(),
        MaskConv2d('B', in_channels // 2, in_channels, 1, **kwargs)
    ])

  def forward(self, x, cond=None):
    out = x
    for layer in self.block:
      if isinstance(layer, MaskConv2d):
        out = layer(out, cond=cond)
      else:
        out = layer(out)
    return out + x

class LayerNorm(nn.LayerNorm):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, x):
    x = x.permute(0, 2, 3, 1).contiguous()
    x_shape = x.shape
    x = super().forward(x)
    return x.permute(0, 3, 1, 2).contiguous()

class PixelCNN(nn.Module):
  def __init__(self, H):
    super().__init__()
    assert H.n_layers >= 2
    self.H = H
    input_shape = [1, 28, 28]
    n_channels = input_shape[0]

    if H.use_resblock:
      def block_init(): return ResBlock(H.n_filters)
    else:
      def block_init(): return MaskConv2d('B', H.n_filters, H.n_filters, kernel_size=H.kernel_size, padding=H.kernel_size // 2)

    model = nn.ModuleList([MaskConv2d('A', n_channels, H.n_filters, kernel_size=H.kernel_size, padding=H.kernel_size // 2)])
    for _ in range(H.n_layers):
      model.append(LayerNorm(H.n_filters))
      model.extend([nn.ReLU(), block_init()])
    model.extend([nn.ReLU(), MaskConv2d('B', H.n_filters, H.n_filters, 1)])
    model.extend([nn.ReLU(), MaskConv2d('B', H.n_filters, n_channels, 1)])
    self.net = model
    self.input_shape = input_shape
    self.n_channels = n_channels

  def forward(self, x):
    batch_size = x.shape[0]
    x = x
    for layer in self.net:
      if isinstance(layer, MaskConv2d) or isinstance(layer, ResBlock):
        x = layer(x)
      else:
        x = layer(x)
    return x.view(batch_size, *self.input_shape)

  def nll(self, x):
    x = x[0]
    out = self.forward(x)
    return F.binary_cross_entropy_with_logits(out, x)
    # return F.cross_entropy(self.forward(x), x.long())

  def sample(self, n):
    samples = torch.zeros(n, *self.input_shape).cuda()
    imgs = []
    with torch.no_grad():
      for r in range(self.input_shape[1]):
        for c in range(self.input_shape[2]):
          logits = self.forward(samples)[:, :, r, c]
          probs = torch.sigmoid(logits)
          samples[:, :, r, c] = torch.bernoulli(probs)
          imgs += [samples.cpu()]
    imgs = np.stack([img.numpy() for img in imgs], axis=1)
    return samples.cpu(), imgs


if __name__ == '__main__':
  # TODO: use low beta 0.1
  # TODO: make network bigger
  from gms.utils import load_mnist
  H = utils.parseC(H)
  writer = SummaryWriter(H.logdir)
  logger = utils.dump_logger({}, writer, 0, H)
  train_ds, test_ds = load_mnist(H.bs)
  _batch = next(iter(train_ds))
  _batch[0] = _batch[0].to(H.device)
  model = PixelCNN(H).to(H.device)
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
    samples, gen = model.sample(10)
    writer.add_video('sampling_process', gen, i, fps=60)
    utils.plot_samples(writer, i, batch[0][:10], samples)
    writer.flush()
    model.train()

  for i in count():
    train_epoch()
    eval()
    logger = utils.dump_logger(logger, writer, i, H)

    if i >= H.done_n:
      break
