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

C = utils.AttrDict()
C.bs = 32
C.z_size = 128
C.bn = 0
C.device = 'cuda'
C.log_n = 1000
C.done_n = 20
C.b = 0.1
C.logdir = './logs/'
C.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
C.lr = 1e-3
C.class_cond = 0
C.hidden_size = 512
C.append_loc = 1
C.overfit_batch = 0
C.n_layer = 2
C.n_head = 4
C.n_embed = 128
C.block_size = 28*28

# TODO: record bits/dim
# TODO: try interpolation
# TODO: barebon, no residual block version

def append_location(x):
  """add xy coords to every pixel"""
  XY = torch.stack(torch.meshgrid(torch.linspace(0, 1, x.shape[-2]), torch.linspace(0, 1, x.shape[-1])), 0)
  return torch.cat([x, XY[None].repeat_interleave(x.shape[0], 0).to(x.device)], 1)

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


class GPT(nn.Module):
  """  the full GPT language model, with a context size of block_size """
  def __init__(self, C):
    super().__init__()
    # input embedding stem
    self.pixel_emb = nn.Conv2d(3, C.n_embed, kernel_size=1, stride=1)
    # transformer
    self.blocks = nn.Sequential(*[Block(C) for _ in range(C.n_layer)])
    # decoder head
    self.ln_f = nn.LayerNorm(C.n_embed)
    self.head = nn.Conv2d(C.n_embed, 1, kernel_size=1, stride=1, bias=False)
    #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
    self.C = C

  def forward(self, x):
    batch_size = x.shape[0]
    x = append_location(x)
    # forward the GPT model
    x = self.pixel_emb(x)
    x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, 28*28, -1)
    # add padding on left so that we can't see ourself.
    x = torch.cat([torch.zeros(batch_size, 1, self.C.n_embed).to(self.C.device), x[:, :-1]], dim=1)
    x = self.blocks(x)
    x = self.ln_f(x)
    x = x.permute(0, 2, 1).view(batch_size, -1, 28, 28)
    x = self.head(x)
    return x

  def nll(self, x):
    x = x[0]
    logits = self.forward(x)
    return F.binary_cross_entropy_with_logits(logits, x)

  def sample(self, n):
    imgs = []
    with torch.no_grad():
      samples = torch.zeros(n, 1, 28, 28).to(self.C.device)
      for r in range(28):
        for c in range(28):
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
  C = utils.parseC(C)
  writer = SummaryWriter(C.logdir)
  logger = utils.dump_logger({}, writer, 0, C)
  train_ds, test_ds = load_mnist(C.bs)
  _batch = next(iter(train_ds))
  _batch[0] = _batch[0].to(C.device)
  model = GPT(C).to(C.device)
  optimizer = Adam(model.parameters(), lr=C.lr)

  def train_epoch():
    if C.overfit_batch:
      for i in range(C.log_n):
        optimizer.zero_grad()
        loss = model.nll(_batch)
        loss.backward()
        optimizer.step()
        logger['loss'] += [loss.detach().cpu()]
    else:
      for batch in train_ds:
        batch[0], batch[1] = batch[0].to(C.device), batch[1].to(C.device)
        optimizer.zero_grad()
        loss = model.nll(batch)
        loss.backward()
        optimizer.step()
        logger['loss'] += [loss.detach().cpu()]

  def eval():
    model.eval()
    if C.overfit_batch:
      batch = _batch
      loss = model.nll(batch)
      logger['test/bits_per_dim'] = loss.item() / np.log(2)
    else:
      total_loss = 0
      with torch.no_grad():
        for batch in test_ds:
          batch[0], batch[1] = batch[0].to(C.device), batch[1].to(C.device)
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
    logger = utils.dump_logger(logger, writer, i, C)

    if i >= C.done_n:
      break
