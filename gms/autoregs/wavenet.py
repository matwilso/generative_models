import torch
import torch.nn.functional as F
from torch import distributions as tdib
from torch import nn
from torch.optim import Adam

from gms import common

# Part of implementation pulled from https://github.com/rll/deepul/blob/master/demos/lecture2_autoregressive_models_demos.ipynb,
# which originally pulled from https://github.com/ryujaehun/wavenet


class Wavenet(common.Autoreg):
    """
    This applies the idea of Wavenet and to a 1d-ified MNIST.
    """

    DG = common.AttrDict()
    DG.use_resblock = 1
    DG.hidden_size = 320

    def __init__(self, G):
        super().__init__(G)
        in_channels = 3  # pixel + xy location
        out_channels = 1  # pixel
        res_channels = G.hidden_size
        layer_size = 9  # Largest dilation is 512 (2**9)
        self.causal = DilatedCausalConv1d('A', in_channels, res_channels, dilation=1)
        if G.use_resblock:
            self.stack = nn.Sequential(
                *[ResidualBlock(res_channels, 2**i) for i in range(layer_size)]
            )
        else:
            self.stack = nn.Sequential(
                *[
                    DilatedCausalConv1d('B', res_channels, res_channels, 2**i)
                    for i in range(layer_size)
                ]
            )
        self.out_conv = nn.Conv1d(res_channels, out_channels, 1)
        self.optimizer = Adam(self.parameters(), lr=G.lr)

    def forward(self, x):
        bs = x.shape[0]
        x = common.append_location(x)
        x = x.reshape(bs, -1, 784)
        x = self.causal(x)
        x = self.stack(x)
        x = self.out_conv(x)
        dist = tdib.Bernoulli(logits=x.reshape(bs, 1, 28, 28))
        return dist

    def loss(self, x, y=None):
        dist = self.forward(x)
        loss = -dist.log_prob(x).mean()
        return loss, {'nlogp': loss}

    def sample(self, n):
        steps = []
        batch = torch.zeros(n, 1, 28, 28).to(self.G.device)
        for r in range(28):
            for c in range(28):
                dist = self.forward(batch)
                batch[..., r, c] = dist.sample()[..., r, c]
                steps += [batch.cpu().view(25, 1, 28, 28)]
        return steps[-1], torch.stack(steps)


# Type 'B' Conv
class DilatedCausalConv1d(nn.Module):
    """Dilated Causal Convolution for WaveNet"""

    def __init__(self, mask_type, in_channels, out_channels, dilation=1):
        super(DilatedCausalConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=2, dilation=dilation, padding=0
        )
        self.dilation = dilation
        self.mask_type = mask_type
        assert mask_type in ['A', 'B']

    def forward(self, x):
        if self.mask_type == 'A':
            # you need to pad by 2 here, else you are seeing yourself in the output.
            # 1st one doen't see anything. Nth one sees only n-1.
            return self.conv(F.pad(x, [2, 0]))[..., :-1]
        else:
            # then from then on out, pad as much as you dilate. look at past samples
            return self.conv(F.pad(x, [self.dilation, 0]))


class ResidualBlock(nn.Module):
    def __init__(self, res_channels, dilation):
        super(ResidualBlock, self).__init__()
        # the key Wavenet causal thing is just the structure of the dilation, making sure you only see the past not the future
        self.dilated = DilatedCausalConv1d(
            'B', res_channels, 2 * res_channels, dilation=dilation
        )
        self.conv_res = nn.Conv1d(res_channels, res_channels, 1)

    def forward(self, x):
        output = self.dilated(x)
        # PixelCNN gate
        o1, o2 = output.chunk(2, dim=1)
        output = torch.tanh(o1) * torch.sigmoid(o2)
        output = x + self.conv_res(output)
        return output
