import numpy as np
from torch.optim import Adam
import torch as th
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from gms import utils

class Wavenet(utils.Autoreg):
  """This is basically just taking the idea of Wavenet and applying it to a 1d-ified MNIST.
  That's pretty much it.
  """
  DC = utils.AttrDict()
  DC.use_resblock = 1
  DC.hidden_size = 320
  def __init__(self, C):
    super().__init__(C)
    in_channels = 3 # pixel + xy location
    out_channels = 1 # pixel
    res_channels = C.hidden_size
    layer_size = 9  # Largest dilation is 512 (2**9)
    self.causal = DilatedCausalConv1d('A', in_channels, res_channels, dilation=1)
    if C.use_resblock:
      self.stack = nn.Sequential(*[ResidualBlock(res_channels, 2 ** i) for i in range(layer_size)])
    else:
      self.stack = nn.Sequential(*[DilatedCausalConv1d('B', res_channels, res_channels, 2 ** i) for i in range(layer_size)])
    self.out_conv = nn.Conv1d(res_channels, out_channels, 1)
    self.optimizer = Adam(self.parameters(), lr=C.lr)

  def forward(self, x):
    bs = x.shape[0]
    x = utils.append_location(x)
    x = x.reshape(bs, -1, 784)
    x = self.causal(x)
    x = self.stack(x)
    x = self.out_conv(x)
    dist = tdib.Bernoulli(logits=x.reshape(bs, 1, 28, 28))
    return dist

  def loss(self, x):
    dist = self.forward(x)
    loss = -dist.log_prob(x).mean()
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

# Implementation pulled from https://github.com/rll/deepul/blob/master/demos/lecture2_autoregressive_models_demos.ipynb,
# which originally pulled from https://github.com/ryujaehun/wavenet 
# Type 'B' Conv
class DilatedCausalConv1d(nn.Module):
  """Dilated Causal Convolution for WaveNet"""

  def __init__(self, mask_type, in_channels, out_channels, dilation=1):
    super(DilatedCausalConv1d, self).__init__()
    self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=2, dilation=dilation, padding=0)
    self.dilation = dilation
    self.mask_type = mask_type
    assert mask_type in ['A', 'B']

  def forward(self, x):
    if self.mask_type == 'A':
      # why pad by 2 here? else you are seeing yourself in the output.
      # this ensures the outputs don't see themselves. 1st one doen't see anything. Nth one sees only n-1.
      return self.conv(F.pad(x, [2, 0]))[..., :-1]
    else:
      # then from then on out, pad as much as you dilate. look at past samples
      return self.conv(F.pad(x, [self.dilation, 0]))

class ResidualBlock(nn.Module):
  def __init__(self, res_channels, dilation):
    super(ResidualBlock, self).__init__()
    # these blocks are somewhat distracting from understanding.
    # the key Wavenet causal thing is just the structure of the dilation, making sure you only see the past not the future
    self.dilated = DilatedCausalConv1d('B', res_channels, 2 * res_channels, dilation=dilation)
    self.conv_res = nn.Conv1d(res_channels, res_channels, 1)  # does this really help much, just transforming the data elementwise? i guess you can learn something that you want to just apply on every element. kind of like the forwards in the transformerr

  def forward(self, x):
    output = self.dilated(x)
    # PixelCNN gate
    o1, o2 = output.chunk(2, dim=1)
    output = th.tanh(o1) * th.sigmoid(o2)
    output = x + self.conv_res(output)  # Residual network
    return output