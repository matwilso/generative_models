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

from gms import utils
from og_pixel_cnn import LayerNorm, MaskConv2d

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

class StackLayerNorm(nn.Module):
  def __init__(self, n_filters):
    super().__init__()
    self.h_layer_norm = LayerNorm(n_filters)
    self.v_layer_norm = LayerNorm(n_filters)

  def forward(self, x):
    vx, hx = x.chunk(2, dim=1)
    vx, hx = self.v_layer_norm(vx), self.h_layer_norm(hx)
    return torch.cat((vx, hx), dim=1)

class GatedConv2d(nn.Module):
  def __init__(self, mask_type, in_channels, out_channels, k=7, padding=3):
    super().__init__()

    self.vertical = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=k, padding=padding, bias=False)
    self.horizontal = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(1, k), padding=(0, padding), bias=False)
    self.vtoh = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1, bias=False)
    self.htoh = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    self.register_buffer('vmask', self.vertical.weight.data.clone())
    self.register_buffer('hmask', self.horizontal.weight.data.clone())

    self.vmask.fill_(1)
    self.hmask.fill_(1)

    # zero the bottom half rows of the vmask
    self.vmask[:, :, k // 2 + 1:, :] = 0

    # zero the right half of the hmask
    self.hmask[:, :, :, k // 2 + 1:] = 0
    if mask_type == 'A':
      self.hmask[:, :, :, k // 2] = 0

  def down_shift(self, x):
    x = x[:, :, :-1, :]
    pad = nn.ZeroPad2d((0, 0, 1, 0))
    return pad(x)

  def forward(self, x):
    vx, hx = x.chunk(2, dim=1)

    self.vertical.weight.data *= self.vmask
    self.horizontal.weight.data *= self.hmask

    vx = self.vertical(vx)
    hx_new = self.horizontal(hx)
    # Allow horizontal stack to see information from vertical stack
    hx_new = hx_new + self.vtoh(self.down_shift(vx))

    # Gates
    vx_1, vx_2 = vx.chunk(2, dim=1)
    vx = torch.tanh(vx_1) * torch.sigmoid(vx_2)

    hx_1, hx_2 = hx_new.chunk(2, dim=1)
    hx_new = torch.tanh(hx_1) * torch.sigmoid(hx_2)
    hx_new = self.htoh(hx_new)
    hx = hx + hx_new

    return torch.cat((vx, hx), dim=1)

# GatedPixelCNN using horizontal and vertical stacks to fix blind-spot
class GatedPixelCNN(nn.Module):
  def __init__(self, H, n_layers=5, n_filters=120):
    super().__init__()
    self.H = H
    input_shape = (1,28,28)
    self.n_channels = input_shape[0]
    self.input_shape = input_shape
    self.in_conv = MaskConv2d('A', self.n_channels, n_filters, 7, padding=3)
    model = []
    for _ in range(n_layers - 2):
      model.extend([nn.ReLU(), GatedConv2d('B', n_filters, n_filters, 7, padding=3)])
      model.append(StackLayerNorm(n_filters))
    self.out_conv = MaskConv2d('B', n_filters, self.n_channels, 7, padding=3)
    self.net = nn.Sequential(*model)

  def forward(self, x):
    batch_size = x.shape[0]
    x = x
    x = self.in_conv(x)
    x = self.net(torch.cat((x, x), dim=1)).chunk(2, dim=1)[1]
    x = self.out_conv(x)
    return x.view(batch_size, *self.input_shape)

  def nll(self, x):
    x = x[0]
    return F.binary_cross_entropy_with_logits(self.forward(x), x)

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