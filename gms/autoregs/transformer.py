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

# This transformer code is taken from https://github.com/karpathy/minGPT and then modified.

class TransformerCNN(nn.Module):
  DC = utils.AttrDict()
  DC.n_layer = 2
  DC.n_head = 4
  DC.n_embed = 128
  """  the full GPT language model, with a context size of block_size """
  def __init__(self, in_size=1, block_size=28*28, C=None):
    super().__init__()
    assert C is not None, 'must pass in C'
    self.block_size = block_size
    self.in_size = in_size
    self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, C.n_embed))
    self.embed = nn.Linear(self.in_size, C.n_embed, bias=False)
    self.blocks = nn.Sequential(*[Block(self.block_size, C) for _ in range(C.n_layer)])
    self.ln_f = nn.LayerNorm(C.n_embed)
    self.dist_head = CategoricalHead(C.n_embed, self.in_size, C)
    self.C = C
    self.optimizer = Adam(self.parameters(), lr=self.C.lr)

  def train_step(self, batch):
    x = batch[0].flatten(-2).permute(0, 2, 1)
    self.optimizer.zero_grad()
    loss = -self.forward(x).log_prob(x).mean()
    loss.backward()
    self.optimizer.step()
    return {'loss': loss}

  def loss(self, batch):
    return torch.zeros(1), {}

  def forward(self, x):
    BS, T, C = x.shape
    # SHIFT RIGHT (add a padding on the left) so you can't see yourself 
    x = torch.cat([torch.zeros(BS, 1, C).to(self.C.device), x[:, :-1]], dim=1)
    # forward the GPT model
    x = self.embed(x)
    x += self.pos_emb # each position maps to a (learnable) vector
    # add padding on left so that we can't see ourself.
    x = self.blocks(x)
    logits = self.ln_f(x)
    return self.dist_head(logits)

  def evaluate(self, writer, batch, epoch):
    samples, gen = self.model.sample(10)
    writer.add_video('sampling_process', gen, epoch, fps=60)
    utils.plot_samples(writer, epoch, batch[0][:10], samples)

  def sample(self, n):
    steps = []
    with torch.no_grad():
      # sample, but the first in the block will stay zero
      batch = torch.zeros(n, self.block_size, self.in_size).to(self.C.device)
      for i in range(self.block_size):
        dist = self.forward(batch)
        batch[:,i] = dist.sample()[:,i]
        steps += [batch.cpu()]
    return batch, steps

class CausalSelfAttention(nn.Module):
  """
  A vanilla multi-head masked self-attention layer with a projection at the end.
  It is possible to use torch.nn.MultiheadAttention here but I am including an
  explicit implementation here to show that there is nothing too scary here.
  """
  def __init__(self, block_size, C):
    super().__init__()
    self.block_size = block_size
    assert C.n_embed % C.n_head == 0
    # key, query, value projections for all heads
    self.key = nn.Linear(C.n_embed, C.n_embed)
    self.query = nn.Linear(C.n_embed, C.n_embed)
    self.value = nn.Linear(C.n_embed, C.n_embed)
    # output projection
    self.proj = nn.Linear(C.n_embed, C.n_embed)
    # causal mask to ensure that attention is only applied to the left in the input sequence
    self.register_buffer("mask", torch.tril(torch.ones(self.block_size, self.block_size)).view(1, 1, self.block_size, self.block_size))
    self.C = C

  def forward(self, x, layer_past=None):
    B, T, C = x.size()
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    k = self.key(x).view(B, T, self.C.n_head, C // self.C.n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = self.query(x).view(B, T, self.C.n_head, C // self.C.n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = self.value(x).view(B, T, self.C.n_head, C // self.C.n_head).transpose(1, 2)  # (B, nh, T, hs)
    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
    att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
    # output projection
    y = self.proj(y)
    return y

class Block(nn.Module):
  """ an unassuming Transformer block """
  def __init__(self, block_size, C):
    super().__init__()
    self.ln1 = nn.LayerNorm(C.n_embed)
    self.ln2 = nn.LayerNorm(C.n_embed)
    self.attn = CausalSelfAttention(block_size, C)
    self.mlp = nn.Sequential(
        nn.Linear(C.n_embed, 4 * C.n_embed),
        nn.GELU(),
        nn.Linear(4 * C.n_embed, C.n_embed),
    )
  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x

class CategoricalHead(nn.Module):
  """take logits and produce a multinomial distribution independently"""
  def __init__(self, in_n, out_n, C):
    super().__init__()
    self.C = C
    self.layer = nn.Linear(in_n, out_n)
  def forward(self, x, past_o=None):
    x = self.layer(x)
    return tdib.Multinomial(logits=x)

class BinaryHead(nn.Module):
  """take logits and produce a bernoulli distribution independently"""
  def __init__(self, in_n, out_n, C):
    super().__init__()
    self.C = C
    self.layer = nn.Linear(in_n, out_n)
  def forward(self, x, past_o=None):
    x = self.layer(x)
    return tdib.Bernoulli(logits=x)