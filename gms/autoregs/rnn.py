import numpy as np
import torch
from torch import distributions as tdib
from torch import nn

from gms import common


class RNN(common.Autoreg):
    DG = common.AttrDict()
    DG.append_loc = 1
    DG.hidden_size = 1024  # this is big and it makes it train slowly, but it makes it have similar # parameters as other models.

    def __init__(self, G):
        super().__init__(G)
        self.input_shape = input_shape = (1, 28, 28)
        self.input_channels = input_shape[0] + 2 if G.append_loc else input_shape[0]
        self.canvas_size = input_shape[1] * input_shape[2]
        self.lstm = nn.LSTM(
            self.input_channels, self.G.hidden_size, num_layers=1, batch_first=True
        )
        self.fc = nn.Linear(self.G.hidden_size, input_shape[0])

    def loss(self, inp, y=None):
        bs = inp.shape[0]
        x = common.append_location(inp) if self.G.append_loc else inp

        # make LSTM operate over 1 pixel at a time.
        x = (
            x.permute(0, 2, 3, 1)
            .contiguous()
            .view(bs, self.canvas_size, self.input_channels)
        )
        # align it so we are predicting the next pixel. start with dummy first and feed everything put last real pixel.
        x = torch.cat(
            (torch.zeros(bs, 1, self.input_channels).to(self.G.device), x[:, :-1]),
            dim=1,
        )

        h0 = torch.zeros(1, x.size(0), self.G.hidden_size).to(self.G.device)
        c0 = torch.zeros(1, x.size(0), self.G.hidden_size).to(self.G.device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (bs, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out).squeeze(-1)  # b x 784
        loss = -tdib.Bernoulli(logits=out.reshape([bs, 1, 28, 28])).log_prob(inp).mean()
        return loss, {'nlogp': loss}

    def sample(self, n):
        steps = []
        viz = torch.zeros(n, 1, 28, 28).to(self.G.device)
        with torch.no_grad():
            samples = torch.zeros(n, 1, self.input_channels).to(self.G.device)
            G = torch.zeros(1, n, self.G.hidden_size).to(self.G.device)
            c = torch.zeros(1, n, self.G.hidden_size).to(self.G.device)

            for i in range(self.canvas_size):
                x_inp = samples[:, [i]]
                out, (G, c) = self.lstm(x_inp, (G, c))
                out = self.fc(out[:, 0, :])
                prob = torch.sigmoid(out)
                sample_pixel = torch.bernoulli(prob).unsqueeze(-1)  # n x 1 x 1
                if self.G.append_loc:
                    loc = np.array([i // 28, i % 28]) / 27
                    loc = torch.FloatTensor(loc).to(self.G.device)
                    loc = loc.view(1, 1, 2).repeat(n, 1, 1)
                    sample_pixel = torch.cat((sample_pixel, loc), dim=-1)
                samples = torch.cat((samples, sample_pixel), dim=1)
                viz[:, :, i // 28, i % 28] = sample_pixel[:, :, 0]
                steps += [viz.clone()]
            samples = (
                samples[:, 1:, 0] if self.G.append_loc else samples[:, 1:].squeeze(-1)
            )
            samples = samples.view(n, *self.input_shape)
            return samples.cpu(), torch.stack(steps)
