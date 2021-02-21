import math
import torch
from torch import nn
import torch.functional as F

def get_timestep_embedding(timesteps, embedding_dim: int):
  """
  From Fairseq.
  Build sinusoidal embeddings.
  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".
  """
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

  half_dim = embedding_dim // 2
  emb = math.log(10000) / (half_dim - 1)
  emb = torch.exp(torch.range(half_dim, dtype=DEFAULT_DTYPE) * -emb)
  # emb = torch.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
  emb = torch.cast(timesteps, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
  emb = torch.concat([torch.sin(emb), torch.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    # emb = torch.concat([emb, torch.zeros([num_embeddings, 1])], axis=1)
    emb = torch.pad(emb, [[0, 0], [0, 1]])
  assert emb.shape == [timesteps.shape[0], embedding_dim]
  return emb

def swish(x):
  return x * torch.sigmoid(x)

class TembNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.temb_net = nn.Sequential(
      nn.Linear(C, ch*4),
      Swish(),
      nn.Linear(ch*4, ch*4)
    )
  def forward(t):
    temb = get_timestep_embedding(t, ch)
    self.temb_net(t)

class BasicNet(nn.Module):
  def __init__(self, C):
    super().__init__()
    self.c1 = nn.Conv2d(1, 128, 7, 1, padding=3)
    self.c2 = nn.Conv2d(128, 128, 7, 1, padding=3)
    self.c3 = nn.Conv2d(128, 128, 3, 1, padding=1)
    self.c4 = nn.Conv2d(128, 128, 3, 1, padding=1)
    self.cout = nn.Conv2d(128, 2, 3, 1, padding=1)
    #self.gn1 = nn.GroupNorm(32, 128)
    #self.gn2 = nn.GroupNorm(32, 128)

  def forward(self, x, t):
    x = self.c1(x)
    x = swish(x)
    x = xp2 = x + self.temb(t)[:,None,None,:]
    x = self.c2(x)
    x = swish(x)
    x = self.c3(x)
    x = xp4 = swish(x + xp2)
    #x = xp4 = swish(self.gn1(x + xp2))
    x = self.c4(x)
    x = swish(x)
    x = self.cout(x)
    return x