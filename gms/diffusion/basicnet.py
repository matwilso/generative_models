import math
import torch as th
from torch import nn
# arch maintains same shape, has resnet skips, and injects the time embedding in many places

class SiLU(nn.Module):
  def forward(self, x):
    return x * th.sigmoid(x)

def zero_module(module):
  """
  Zero out the parameters of a module and return it.
  """
  for p in module.parameters():
    p.detach().zero_()
  return module

def scale_module(module, scale):
  """
  Scale the parameters of a module and return it.
  """
  for p in module.parameters():
    p.detach().mul_(scale)
  return module

def mean_flat(tensor):
  """
  Take the mean over all non-batch dimensions.
  """
  return tensor.mean(dim=list(range(1, len(tensor.shape))))

def normalization(channels):
  """
  Make a standard normalization layer.

  :param channels: number of input channels.
  :return: an nn.Module for normalization.
  """
  return nn.GroupNorm(32, channels)

def timestep_embedding(timesteps, dim, max_period=500):
  """
  Create sinusoidal timestep embeddings.

  :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
  :param dim: the dimension of the output.
  :param max_period: controls the minimum frequency of the embeddings.
  :return: an [N x dim] Tensor of positional embeddings.
  """
  half = dim // 2
  freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
  args = timesteps[:, None].float() * freqs[None]
  embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
  if dim % 2:
    embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
  return embedding

class ResBlock(nn.Module):
  def __init__(self, channels, emb_channels):
    super().__init__()
    self.channels = channels
    self.emb_channels = emb_channels
    self.in_layers = nn.Sequential(
        normalization(channels),
        SiLU(),
        nn.Conv2d(channels, channels, 3, padding=1),
    )
    self.emb_layers = nn.Sequential(
        SiLU(),
        nn.Linear(emb_channels, channels),
    )
    self.out_layers = nn.Sequential(
        normalization(channels),
        SiLU(),
        zero_module(nn.Conv2d(channels, channels, 3, padding=1)),
    )

  def forward(self, x, emb):
    """
    Apply the block to a Tensor, conditioned on a timestep embedding.

    :param x: an [N x C x ...] Tensor of features.
    :param emb: an [N x emb_channels] Tensor of timestep embeddings.
    :return: an [N x C x ...] Tensor of outputs.
    """
    h = self.in_layers(x)
    emb_out = self.emb_layers(emb).type(h.dtype)[...,None,None]
    h = h + emb_out
    h = self.out_layers(h)
    return x + h

class BasicNet(nn.Module):
  def __init__(self, C):
    super().__init__()
    time_embed_dim = 64 * 4
    self.cin = nn.Conv2d(1, C.hidden_size, 3, padding=1)
    self.r1 = ResBlock(C.hidden_size, time_embed_dim)
    self.r2 = ResBlock(C.hidden_size, time_embed_dim)
    self.r3 = ResBlock(C.hidden_size, time_embed_dim)
    self.cout = nn.Conv2d(C.hidden_size, 2, 3, padding=1)

    self.time_embed = nn.Sequential(
        nn.Linear(64, time_embed_dim),
        SiLU(),
        nn.Linear(time_embed_dim, time_embed_dim),
    )

  def forward(self, x, timesteps):
    emb = self.time_embed(timestep_embedding(timesteps.float(), 64))
    x = self.cin(x)
    x = self.r1(x, emb)
    x = self.r2(x, emb)
    x = self.r3(x, emb)
    x = self.cout(x)
    return x