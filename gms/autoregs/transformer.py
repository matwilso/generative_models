import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from gms import common

# This transformer code is taken from https://github.com/karpathy/minGPT and then modified.


class TransformerCNN(common.Autoreg):
    DG = common.AttrDict()
    DG.n_layer = 2
    DG.n_head = 4
    DG.n_embed = 128
    DG.lr = 1e-3
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, in_size=1, block_size=28 * 28, head='bin', G=None):
        super().__init__(G)
        assert G is not None, 'must pass in G'
        self.block_size = block_size
        self.in_size = in_size
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.block_size, G.n_embed)
        )  # learned position embedding
        self.embed = nn.Linear(self.in_size, G.n_embed, bias=False)
        self.blocks = nn.Sequential(
            *[Block(self.block_size, G) for _ in range(G.n_layer)]
        )
        self.ln_f = nn.LayerNorm(G.n_embed)
        if head == 'bin':
            self.dist_head = common.BinaryHead(G.n_embed, self.in_size, G)
        elif head == 'cat':
            self.dist_head = common.CategoricalHead(G.n_embed, self.in_size, G)
        self.optimizer = Adam(self.parameters(), lr=self.G.lr)

    def train_step(self, x):
        x = x.flatten(-2).permute(0, 2, 1)
        self.optimizer.zero_grad()
        loss = -self.forward(x).log_prob(x).mean()
        loss.backward()
        self.optimizer.step()
        return {'nlogp': loss}

    def forward(self, x):
        BS, _, G = x.shape
        # SHIFT RIGHT (add a padding on the left) so you can't see yourself
        x = torch.cat([torch.zeros(BS, 1, G).to(self.G.device), x[:, :-1]], dim=1)
        # forward the GPT model
        x = self.embed(x)
        x += self.pos_emb  # each position maps to a (learnable) vector
        # add padding on left so that we can't see ourself.
        x = self.blocks(x)
        logits = self.ln_f(x)
        return self.dist_head(logits)

    def sample(self, n):
        steps = []
        batch = torch.zeros(n, self.block_size, self.in_size).to(self.G.device)
        for i in range(self.block_size):
            dist = self.forward(batch)
            batch[:, i] = dist.sample()[:, i]
            steps += [batch.cpu()]
        return batch, steps

    def evaluate(self, writer, x, epoch):
        samples, gen = self.sample(25)
        B, HW, C = samples.shape
        gen = torch.stack(gen).reshape([HW, B, 1, 28, 28]).permute(1, 0, 2, 3, 4)
        samples = samples.reshape([B, C, 28, 28]).cpu()
        writer.add_video(
            'sampling_process',
            common.combine_imgs(gen, 5, 5)[None, :, None],
            epoch,
            fps=60,
        )
        writer.add_image('samples', common.combine_imgs(samples, 5, 5)[None], epoch)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use  torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, block_size, G):
        super().__init__()
        self.block_size = block_size
        assert G.n_embed % G.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(G.n_embed, G.n_embed)
        self.query = nn.Linear(G.n_embed, G.n_embed)
        self.value = nn.Linear(G.n_embed, G.n_embed)
        # output projection
        self.proj = nn.Linear(G.n_embed, G.n_embed)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.block_size, self.block_size)).view(
                1, 1, self.block_size, self.block_size
            ),
        )
        self.n_head = G.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.proj(y)
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, block_size, G):
        super().__init__()
        self.ln1 = nn.LayerNorm(G.n_embed)
        self.ln2 = nn.LayerNorm(G.n_embed)
        self.attn = CausalSelfAttention(block_size, G)
        self.mlp = nn.Sequential(
            nn.Linear(G.n_embed, 4 * G.n_embed),
            nn.GELU(),
            nn.Linear(4 * G.n_embed, G.n_embed),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
