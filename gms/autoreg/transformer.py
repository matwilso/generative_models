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
H.lr = 1e-3
H.class_cond = 0
H.hidden_size = 512
H.append_loc = 1
H.overfit_batch = 0
H.n_layer = 2
H.n_head = 4
H.n_embed = 128
H.block_size = 28*28

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

  def __init__(self, H):
    super().__init__()
    assert H.n_embed % H.n_head == 0
    # key, query, value projections for all heads
    self.key = nn.Linear(H.n_embed, H.n_embed)
    self.query = nn.Linear(H.n_embed, H.n_embed)
    self.value = nn.Linear(H.n_embed, H.n_embed)
    # output projection
    self.proj = nn.Linear(H.n_embed, H.n_embed)
    # causal mask to ensure that attention is only applied to the left in the input sequence
    self.register_buffer("mask", torch.tril(torch.ones(H.block_size, H.block_size)).view(1, 1, H.block_size, H.block_size))
    self.H = H

  def forward(self, x, layer_past=None):
    B, T, C = x.size()
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    k = self.key(x).view(B, T, self.H.n_head, C // self.H.n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = self.query(x).view(B, T, self.H.n_head, C // self.H.n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = self.value(x).view(B, T, self.H.n_head, C // self.H.n_head).transpose(1, 2)  # (B, nh, T, hs)
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
  def __init__(self, H):
    super().__init__()
    self.ln1 = nn.LayerNorm(H.n_embed)
    self.ln2 = nn.LayerNorm(H.n_embed)
    self.attn = CausalSelfAttention(H)
    self.mlp = nn.Sequential(
        nn.Linear(H.n_embed, 4 * H.n_embed),
        nn.GELU(),
        nn.Linear(4 * H.n_embed, H.n_embed),
    )

  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x

class GPT(nn.Module):
  """  the full GPT language model, with a context size of block_size """
  def __init__(self, H):
    super().__init__()
    # input embedding stem
    self.pixel_emb = nn.Conv2d(3, H.n_embed, kernel_size=1, stride=1)
    # transformer
    self.blocks = nn.Sequential(*[Block(H) for _ in range(H.n_layer)])
    # decoder head
    self.ln_f = nn.LayerNorm(H.n_embed)
    self.head = nn.Conv2d(H.n_embed, 1, kernel_size=1, stride=1, bias=False)
    #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
    self.H = H

  def forward(self, x):
    batch_size = x.shape[0]
    x = append_location(x)
    # forward the GPT model
    x = self.pixel_emb(x)
    x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, 28*28, -1)
    # add padding on left so that we can't see ourself.
    x = torch.cat([torch.zeros(batch_size, 1, self.H.n_embed).to(self.H.device), x[:, :-1]], dim=1)
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
      samples = torch.zeros(n, 1, 28, 28).to(self.H.device)
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
  H = utils.parseH(H)
  writer = SummaryWriter(H.logdir)
  logger = utils.dump_logger({}, writer, 0, H)
  train_ds, test_ds = load_mnist(H.bs)
  _batch = next(iter(train_ds))
  _batch[0] = _batch[0].to(H.device)
  model = GPT(H).to(H.device)
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
