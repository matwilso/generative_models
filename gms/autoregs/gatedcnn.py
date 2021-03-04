import numpy as np
from torch.optim import Adam
import torch as th
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from gms import utils
from gms.autoregs.pixelcnn import PixelCNN, LayerNorm, MaskConv2d

# GatedPixelCNN using horizontal and vertical stacks to fix blind-spot
class GatedPixelCNN(PixelCNN):
  DC = utils.AttrDict()
  DC.n_filters = 96
  DC.n_layers = 5
  DC.kernel_size = 7
  DC.use_resblock = 0

  def __init__(self, C):
    super().__init__(C)
    input_shape = (1,28,28)
    self.n_channels = input_shape[0]
    self.input_shape = input_shape
    self.in_conv = MaskConv2d('A', self.n_channels, C.n_filters, 7, padding=3)
    model = []
    for _ in range(C.n_layers - 2):
      model.extend([nn.ReLU(), GatedConv2d('B', C.n_filters, C.n_filters, 7, padding=3)])
      model.append(StackLayerNorm(C.n_filters))
    self.out_conv = MaskConv2d('B', C.n_filters, self.n_channels, 7, padding=3)
    self.net = nn.Sequential(*model)

  def forward(self, x):
    batch_size = x.shape[0]
    x = x
    x = self.in_conv(x)
    x = self.net(th.cat((x, x), dim=1)).chunk(2, dim=1)[1]
    x = self.out_conv(x)
    return tdib.Bernoulli(logits=x.view(batch_size, *self.input_shape))

class StackLayerNorm(nn.Module):
  def __init__(self, n_filters):
    super().__init__()
    self.h_layer_norm = LayerNorm(n_filters)
    self.v_layer_norm = LayerNorm(n_filters)

  def forward(self, x):
    vx, hx = x.chunk(2, dim=1)
    vx, hx = self.v_layer_norm(vx), self.h_layer_norm(hx)
    return th.cat((vx, hx), dim=1)

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
    vx = th.tanh(vx_1) * th.sigmoid(vx_2)

    hx_1, hx_2 = hx_new.chunk(2, dim=1)
    hx_new = th.tanh(hx_1) * th.sigmoid(hx_2)
    hx_new = self.htoh(hx_new)
    hx = hx + hx_new

    return th.cat((vx, hx), dim=1)

