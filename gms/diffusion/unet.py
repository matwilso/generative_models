import torch
from torch import nn
import torch.functional as F

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

    x += nn.Linear(C, out_ch)(swish(temb))[:, None, None, :]

    # add in timestep embedding
    h += nn.dense(swish(temb), name='temb_proj', num_units=out_ch)[:, None, None, :]

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

class HoUnet(nn.Module):
  def __init__(self, C):
    super().__init__()
    # model(x, *, t, y, name, num_classes, reuse=tf.AUTO_REUSE, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0., resamp_with_conv=True):
    self._middle = nn.Sequential(
      ResnetBlock(),
      SelfAttention(),
      ResnetBlock(),
    )

  def forward(x):
    B, S, _, _ = x.shape
    num_resolutions = len(ch_mult)

    assert num_classes == 1 and y is None, 'not supported'
    del y

    # Timestep embedding
    temb = nn.get_timestep_embedding(t, ch)
    temb = nn.dense(temb, name='dense0', num_units=ch * 4)
    temb = nn.dense(swish(temb), name='dense1', num_units=ch * 4)
    assert temb.shape == [B, ch * 4]

    # Downsampling
    hs = [nn.conv2d(x, name='conv_in', num_units=ch)]
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        h = resnet_block(
            hs[-1], name='block_{}'.format(i_block), temb=temb, out_ch=ch * ch_mult[i_level], dropout=dropout)
        if h.shape[1] in attn_resolutions:
          h = attn_block(h, name='attn_{}'.format(i_block), temb=temb)
        hs.append(h)
      # Downsample
      if i_level != num_resolutions - 1:
        hs.append(downsample(hs[-1], name='downsample', with_conv=resamp_with_conv))

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
    h = swish(normalize(h, temb=temb, name='norm_out'))
    h = nn.conv2d(h, name='conv_out', num_units=out_ch, init_scale=0.)
    assert h.shape == x.shape[:3] + [out_ch]
    return h