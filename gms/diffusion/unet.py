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
  return x * F.sigmoid(x)

class Swish(nn.Module):
  def forward(self, input: Tensor) -> Tensor:
    return swish(input)

# TODO: probably conver these to sequential models
class Upsample(nn.Module):
  def __init__(self):
    super().__init__()
    self.ups = nn.Upsample(scale_factor=2, mode='nearest', align_corners=True)
    self.conv = nn.Conv2d(C, C, 3, 1)  # TODO: make sure it keeps same size

  def forward(self, x):
    return self.conv(self.ups(x))

class Downsample(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(C, C, 3, 2)  # TODO: make sure it keeps same size

  def forward(self, x):
    return self.conv(x)

class ResnetBlock(nn.Module):
  def __init__(self):
    super().__init__()
    self.gn = nn.GroupNorm(32, C)
    import ipdb; ipdb.set_trace()

  def forward(self, x, temb):
    x = swish(self.gn(x))
    x = nn.Conv2d(C, out_ch)(x)

    # timestep embedding
    x += nn.Linear(C, out_ch)(swish(temb))[:, None, None, :]

    x = swish(self.gn(x))
    x = self.dropout(x)
    x = nn

    h = swish(normalize(h, temb=temb, name='norm2'))
    h = tf.nn.dropout(h, rate=dropout)
    h = nn.conv2d(h, name='conv2', num_units=out_ch, init_scale=0.)

    if C != out_ch:
      if conv_shortcut:
        x = nn.conv2d(x, name='conv_shortcut', num_units=out_ch)
      else:
        x = nn.nin(x, name='nin_shortcut', num_units=out_ch)

    assert x.shape == h.shape
    print('{}: x={} temb={}'.format(tf.get_default_graph().get_name_scope(), x.shape, temb.shape))
    return x + h

class SelfAttention(nn.Module):
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
    self.C = C

  def forward(self, x, layer_past=None):
    B, T, C = x.size()
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    k = self.key(x).view(B, T, self.C.n_head, C // self.C.n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = self.query(x).view(B, T, self.C.n_head, C // self.C.n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = self.value(x).view(B, T, self.C.n_head, C // self.C.n_head).transpose(1, 2)  # (B, nh, T, hs)
    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
    # output projection
    y = self.proj(y)
    return y

class Unet(nn.Module):
  """Special type of Unet from https://arxiv.org/pdf/2006.11239.pdf"""
  def __init__(self, C):
    super().__init__()
    # model(x, *, t, y, name, num_classes, reuse=tf.AUTO_REUSE, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0., resamp_with_conv=True):
    self._middle = nn.Sequential(
      ResnetBlock(),
      SelfAttention(),
      ResnetBlock(),
    )
    self.temb_net = nn.Sequential(
      nn.Linear(C, ch*4),
      Swish(),
      nn.Linear(ch*4, ch*4)
    )

    hs = [nn.conv2d(x, name='conv_in', num_units=ch)]

    self._out_gn = nn.GroupNorm(32, C)
    self._conv_out = nn.Conv2d(C, out_ch, 3, 1) # pad = same

  def forward(x):
    B, S, _, _ = x.shape
    num_resolutions = len(ch_mult)
    # Timestep embedding
    temb = get_timestep_embedding(t, ch)
    temb = self.temb_net(temb)

    # Downsampling
    xs = [self.down1(x)]
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        x = self.resnetI(xs[-1], name='block_{}'.format(i_block), temb=temb, out_ch=ch * ch_mult[i_level], dropout=dropout)
        if h.shape[1] in attn_resolutions:
          h = attn_block(h, name='attn_{}'.format(i_block), temb=temb)
        xs.append(x)
      # Downsample
      if i_level != num_resolutions - 1:
        xs.append(downsample(xs[-1], name='downsample', with_conv=resamp_with_conv))

    # Middle
    h = self._middle(hs[-1])

    # Upsampling
    for i_level in reversed(range(num_resolutions)):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks + 1):
        h = resnet_block(tf.concat([h, hs.pop()], axis=-1), name='block_{}'.format(i_block),
                         temb=temb, out_ch=ch * ch_mult[i_level], dropout=dropout)
        if h.shape[1] in attn_resolutions:
          h = attn_block(h, name='attn_{}'.format(i_block), temb=temb)
      # Upsample
      if i_level != 0:
        h = upsample(h, name='upsample', with_conv=resamp_with_conv)
    assert not hs

    # End
    h = swish(self._out_gn(h))
    h = self.conv_out(H)
    assert h.shape == x.shape[:3] + [out_ch]
    return h

