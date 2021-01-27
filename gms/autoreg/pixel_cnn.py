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

class MaskConv2d(nn.Conv2d):
  def __init__(self, mask_type, *args, conditional_size=None,
               color_conditioning=False, **kwargs):
    assert mask_type == 'A' or mask_type == 'B'
    super().__init__(*args, **kwargs)
    self.conditional_size = conditional_size
    self.color_conditioning = color_conditioning
    self.register_buffer('mask', torch.zeros_like(self.weight))
    self.create_mask(mask_type)
    if self.conditional_size:
      if len(self.conditional_size) == 1:
        self.cond_op = nn.Linear(conditional_size[0], self.out_channels)
      else:
        self.cond_op = nn.Conv2d(conditional_size[0], self.out_channels, kernel_size=3, padding=1)

  def forward(self, input, cond=None):
    batch_size = input.shape[0]
    out = F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                   self.padding, self.dilation, self.groups)
    if self.conditional_size:
      if len(self.conditional_size) == 1:
        # Broadcast across height and width of image and add as conditional bias
        out = out + self.cond_op(cond).view(batch_size, -1, 1, 1)
      else:
        out = out + self.cond_op(cond)
    return out

  def create_mask(self, mask_type):
    k = self.kernel_size[0]
    self.mask[:, :, :k // 2] = 1
    self.mask[:, :, k // 2, :k // 2] = 1
    if self.color_conditioning:
      assert self.in_channels % 3 == 0 and self.out_channels % 3 == 0
      one_third_in, one_third_out = self.in_channels // 3, self.out_channels // 3
      if mask_type == 'B':
        self.mask[:one_third_out, :one_third_in, k // 2, k // 2] = 1
        self.mask[one_third_out:2 * one_third_out, :2 * one_third_in, k // 2, k // 2] = 1
        self.mask[2 * one_third_out:, :, k // 2, k // 2] = 1
      else:
        self.mask[one_third_out:2 * one_third_out, :one_third_in, k // 2, k // 2] = 1
        self.mask[2 * one_third_out:, :2 * one_third_in, k // 2, k // 2] = 1
    else:
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
  def __init__(self, color_conditioning, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.color_conditioning = color_conditioning

  def forward(self, x):
    x = x.permute(0, 2, 3, 1).contiguous()
    x_shape = x.shape
    if self.color_conditioning:
      x = x.contiguous().view(*(x_shape[:-1] + (3, -1)))
    x = super().forward(x)
    if self.color_conditioning:
      x = x.view(*x_shape)
    return x.permute(0, 3, 1, 2).contiguous()

class PixelCNN(nn.Module):
  def __init__(self, input_shape, n_colors, n_filters=64,
               kernel_size=7, n_layers=5,
               conditional_size=None, use_resblock=False,
               color_conditioning=False):
    super().__init__()
    assert n_layers >= 2
    n_channels = input_shape[0]

    kwargs = dict(conditional_size=conditional_size,
                  color_conditioning=color_conditioning)
    if use_resblock:
      def block_init(): return ResBlock(n_filters, **kwargs)
    else:
      def block_init(): return MaskConv2d('B', n_filters, n_filters,
                                          kernel_size=kernel_size,
                                          padding=kernel_size // 2, **kwargs)

    model = nn.ModuleList([MaskConv2d('A', n_channels, n_filters,
                                      kernel_size=kernel_size,
                                      padding=kernel_size // 2, **kwargs)])
    for _ in range(n_layers):
      if color_conditioning:
        model.append(LayerNorm(color_conditioning, n_filters // 3))
      else:
        model.append(LayerNorm(color_conditioning, n_filters))
      model.extend([nn.ReLU(), block_init()])
    model.extend([nn.ReLU(), MaskConv2d('B', n_filters, n_filters, 1, **kwargs)])
    model.extend([nn.ReLU(), MaskConv2d('B', n_filters, n_colors * n_channels, 1, **kwargs)])

    if conditional_size:
      if len(conditional_size) == 1:
        self.cond_op = lambda x: x  # No preprocessing conditional if one hot
      else:
        # For Grayscale PixelCNN (some preprocessing on the binary image)
        self.cond_op = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

    self.net = model
    self.input_shape = input_shape
    self.n_colors = n_colors
    self.n_channels = n_channels
    self.color_conditioning = color_conditioning
    self.conditional_size = conditional_size

  def forward(self, x, cond=None):
    batch_size = x.shape[0]
    out = (x.float() / (self.n_colors - 1) - 0.5) / 0.5
    if self.conditional_size:
      cond = self.cond_op(cond)
    for layer in self.net:
      if isinstance(layer, MaskConv2d) or isinstance(layer, ResBlock):
        out = layer(out, cond=cond)
      else:
        out = layer(out)

    if self.color_conditioning:
      return out.view(batch_size, self.n_channels, self.n_colors,
                      *self.input_shape[1:]).permute(0, 2, 1, 3, 4)
    else:
      return out.view(batch_size, self.n_colors, *self.input_shape)

  def loss(self, x, cond=None):
    return F.cross_entropy(self(x, cond=cond), x.long())

  def sample(self, n, cond=None):
    samples = torch.zeros(n, *self.input_shape).cuda()
    with torch.no_grad():
      for r in range(self.input_shape[1]):
        for c in range(self.input_shape[2]):
          for k in range(self.n_channels):
            logits = self(samples, cond=cond)[:, :, k, r, c]
            probs = F.softmax(logits, dim=1)
            samples[:, k, r, c] = torch.multinomial(probs, 1).squeeze(-1)
    return samples.permute(0, 2, 3, 1).cpu().numpy()


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
