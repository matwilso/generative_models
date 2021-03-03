from abc import abstractmethod
import math
import torch as th
from torch import nn
import torch.nn.functional as F
# arch maintains same shape, has resnet skips, and injects the time embedding in many places

class SeqCache(nn.Sequential):
  """Sequential, but cache all intermediary outputs"""

  def forward(self, x):
    all_outs = [x]
    for module in self:
      x = module(x)
      x = F.silu(x)
      all_outs += [x]
    return all_outs

class SeqFeed(nn.Sequential):
  """Sequential, but feed in cached inputs"""

  def forward(self, x, feed, emb=None):
    for i, module in enumerate(self):
      x = th.cat([x, feed[i]], axis=1)
      x += emb[..., None, None]
      x = module(x)
      x = F.silu(x)
    return x

class BasicNet(nn.Module):
  def __init__(self, C):
    super().__init__()
    H = C.hidden_size
    time_embed_dim = 2 * H
    self.time_embed = nn.Sequential(
        nn.Linear(64, time_embed_dim),
        nn.SiLU(),
        nn.Linear(time_embed_dim, time_embed_dim),
    )
    self.cin = nn.Conv2d(1, H, 3, padding=1)
    self.down = SeqCache(
        nn.Conv2d(H, H, 3, stride=2, padding=1),
        nn.Conv2d(H, H, 3, stride=2, padding=1),
    )
    self.turn = nn.ConvTranspose2d(H, H, 4, stride=2, padding=1)
    self.t1 = nn.MultiheadAttention(H, 8)
    self.gn1 = nn.GroupNorm(32, H)
    self.up = SeqFeed(
        nn.ConvTranspose2d(2 * H, H, 4, stride=2, padding=1),
        nn.ConvTranspose2d(2 * H, H, 3, padding=1)
    )
    self.cout = nn.Conv2d(H, 2, 3, padding=1)
    self.C = C

  def forward(self, x, timesteps):
    h = x
    emb = self.time_embed(timestep_embedding(timesteps.float(), 64, self.C.timesteps))
    h = self.cin(h)
    # <UNET>
    h, *hoz_skips = self.down(h)[::-1]
    h = self.gn1(h)
    shape = h.shape
    h = h.flatten(-2).permute(2, 0, 1)
    h = self.t1(h, h, h)[0].permute(1, 2, 0).reshape(shape)
    h = self.turn(h)
    h = self.up(h, hoz_skips, emb)
    # </UNET>
    h = self.cout(h)
    return h

def zero_module(module):
  """Zero out the parameters of a module and return it."""
  for p in module.parameters():
    p.detach().zero_()
  return module

def timestep_embedding(timesteps, dim, max_period):
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
