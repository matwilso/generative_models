import numpy as np
from torch.optim import Adam
import torch as th
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from gms import utils

class PixelCNN(utils.Autoreg):
  DC = utils.AttrDict()
  DC.n_filters = 128
  DC.n_layers = 5
  DC.kernel_size = 7
  DC.use_resblock = 0
  DC.lr = 1e-4
  def __init__(self, C):
    super().__init__(C)
    assert C.n_layers >= 2
    input_shape = [1, 28, 28]
    n_channels = input_shape[0]

    if C.use_resblock:
      def block_init(): return ResBlock(C.n_filters)
    else:
      def block_init(): return MaskConv2d('B', C.n_filters, C.n_filters, kernel_size=C.kernel_size, padding=C.kernel_size // 2)

    model = nn.ModuleList([MaskConv2d('A', n_channels, C.n_filters, kernel_size=C.kernel_size, padding=C.kernel_size // 2)])
    for _ in range(C.n_layers):
      model.append(LayerNorm(C.n_filters))
      model.extend([nn.ReLU(), block_init()])
    model.extend([nn.ReLU(), MaskConv2d('B', C.n_filters, C.n_filters, 1)])
    model.extend([nn.ReLU(), MaskConv2d('B', C.n_filters, n_channels, 1)])
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
    return tdib.Bernoulli(logits=x.view(batch_size, *self.input_shape))

  def loss(self, x):
    loss = -self.forward(x).log_prob(x).mean()
    return loss, {'nlogp': loss}

  def sample(self, n):
    steps = []
    batch = th.zeros(n, 1, 28, 28).to(self.C.device)
    for r in range(28):
      for c  in range(28):
        dist = self.forward(batch)
        batch[...,r,c] = dist.sample()[...,r,c]
        steps += [batch.cpu()]
    return batch.cpu(), steps

class MaskConv2d(nn.Conv2d):
  def __init__(self, mask_type, *args, **kwargs):
    assert mask_type == 'A' or mask_type == 'B'
    super().__init__(*args, **kwargs)
    self.register_buffer('mask', th.zeros_like(self.weight))
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