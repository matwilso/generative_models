import numpy as np
import einops
from torch.optim import Adam
import torch as th
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from gms import utils

class TriangularMaskedLinear(nn.Linear):
  def __init__(self, in_size, out_size, bias=True, diagonal=0):
    """
    diagonal: if 0, include self. if -1, exclude self.
    """
    super().__init__(in_size, out_size, bias)
    assert in_size == out_size, 'must be square'
    size = in_size
    # causal mask to ensure that attention is only applied to the left in the input sequence
    self.register_buffer("mask", th.tril(th.ones(size, size), diagonal=diagonal))

  def forward(self, input):
    return F.linear(input, self.mask * self.weight, self.bias)

class MLPBlock(nn.Module): 
  def __init__(self, size, hidden_size=None, triangular=False):
      super().__init__()
      hidden_size = hidden_size or size
      Linear = nn.Linear if not triangular else TriangularMaskedLinear
      self.net = nn.Sequential(
        Linear(size, hidden_size),
        nn.GELU(),
        Linear(hidden_size, size),
      )
  def forward(self, x):
    return self.net(x)

class MixerBlock(nn.Module):
  def __init__(self, temporal_dim, hidden_dim):
    super().__init__()
    # TODO: check what type of triangular we want here
    self.temporal_block = MLPBlock(temporal_dim, triangular=True)
    self.channel_block = MLPBlock(hidden_dim)
    self.ln1 = nn.LayerNorm(hidden_dim)
    self.ln2 = nn.LayerNorm(hidden_dim)

  def forward(self, x):
    y = self.ln1(x)
    y = th.swapaxes(y, 1, 2)
    y = self.temporal_block(y)
    y = th.swapaxes(y, 1, 2)
    x = x + y
    y = self.ln2(x)
    return x + self.channel_block(y)


class Mixer(utils.Autoreg):
  DC = utils.AttrDict()
  DC.n_layer = 2
  DC.n_embed = 64
  DC.lr = 1e-3

  def __init__(self, in_size=1, block_size=28*28, head='bin', C=None):
    super().__init__(C)
    assert C is not None, 'must pass in C'
    #temporal_dim = 28*28
    #img = th.randn(1, 28, 28, 1)
    #position_embedding = th.randn(1, 28, 28, 2)
    #x = th.cat([img, position_embedding], dim=3)
    #x = nn.Linear(3, C.n_embed)(x)

    self.block_size = block_size
    self.in_size = in_size
    self.pos_emb = nn.Parameter(th.zeros(1, self.block_size, C.n_embed)) # learned position embedding
    self.embed = nn.Linear(self.in_size, C.n_embed, bias=False)

    self.blocks = nn.Sequential(*[MixerBlock(self.block_size, C.n_embed) for _ in range(C.n_layer)])

    self.ln_f = nn.LayerNorm(C.n_embed)
    if head == 'bin':
      self.dist_head = utils.BinaryHead(C.n_embed, self.in_size, C)
    elif head == 'cat':
      self.dist_head = utils.CategoricalHead(C.n_embed, self.in_size, C)
    self.optimizer = Adam(self.parameters(), lr=self.C.lr)

  def train_step(self, x):
    x = x.flatten(-2).permute(0, 2, 1)
    self.optimizer.zero_grad()
    loss = -self.forward(x).log_prob(x).mean()
    loss.backward()
    self.optimizer.step()
    return {'nlogp': loss}

  def forward(self, x):
    #x = einops.rearrange(x, 'n h w d -> n (h w) d')
    #x = MlpMixer(temporal_dim, C.n_embed)(x)
    BS, T, C = x.shape
    # SHIFT RIGHT (add a padding on the left) so you can't see yourself 
    x = th.cat([th.zeros(BS, 1, C).to(self.C.device), x[:, :-1]], dim=1)
    # forward the GPT model
    x = self.embed(x)
    x += self.pos_emb # each position maps to a (learnable) vector
    # add padding on left so that we can't see ourself.
    x = self.blocks(x)
    logits = self.ln_f(x)
    return self.dist_head(logits)

  def sample(self, n):
    steps = []
    batch = th.zeros(n, self.block_size, self.in_size).to(self.C.device)
    for i in range(self.block_size):
      dist = self.forward(batch)
      batch[:,i] = dist.sample()[:,i]
      steps += [batch.cpu()]
    return batch, steps

  def evaluate(self, writer, x, epoch):
    samples, gen = self.sample(25)
    B, HW, C = samples.shape
    gen = th.stack(gen).reshape([HW, B, 1, 28, 28]).permute(1, 0, 2, 3, 4)
    samples = samples.reshape([B, C, 28, 28]).cpu()
    writer.add_video('sampling_process', utils.combine_imgs(gen, 5, 5)[None,:,None], epoch, fps=60)
    writer.add_image('samples', utils.combine_imgs(samples, 5, 5)[None], epoch)
