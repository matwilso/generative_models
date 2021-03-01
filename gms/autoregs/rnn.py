import numpy as np
from torch.optim import Adam
import torch as th
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from gms import utils

class RNN(utils.Autoreg):
  DC = utils.AttrDict()
  DC.append_loc = 1
  DC.hidden_size = 1024 # this is big and it makes it train slowly, but it makes it have similar # parameters as other models.
  def __init__(self, C):
    super().__init__(C)
    self.C = C
    self.input_shape = input_shape = (1, 28, 28)
    self.input_channels = input_shape[0] + 2 if C.append_loc else input_shape[0]
    self.canvas_size = input_shape[1] * input_shape[2]
    self.lstm = nn.LSTM(self.input_channels, self.C.hidden_size, num_layers=1, batch_first=True)
    self.fc = nn.Linear(self.C.hidden_size, input_shape[0])

  def loss(self, inp):
    bs = inp.shape[0]
    x = utils.append_location(inp) if self.C.append_loc else inp

    # make LSTM operate over 1 pixel at a time.
    x = x.permute(0, 2, 3, 1).contiguous().view(bs, self.canvas_size, self.input_channels)
    # align it so we are predicting the next pixel. start with dummy first and feed everything put last real pixel.
    x = th.cat((th.zeros(bs, 1, self.input_channels).to(self.C.device), x[:, :-1]), dim=1)

    h0 = th.zeros(1, x.size(0), self.C.hidden_size).to(self.C.device)
    c0 = th.zeros(1, x.size(0), self.C.hidden_size).to(self.C.device)

    # Forward propagate LSTM
    out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (bs, seq_length, hidden_size)

    # Decode the hidden state of the last time step
    out = self.fc(out).squeeze(-1)  # b x 784
    loss = -tdib.Bernoulli(logits=out.reshape([bs, 1, 28, 28])).log_prob(inp).mean()
    return loss, {'nlogp': loss}

  def sample(self, n):
    with th.no_grad():
      samples = th.zeros(n, 1, self.input_channels).to(self.C.device)
      C = th.zeros(1, n, self.C.hidden_size).to(self.C.device)
      c = th.zeros(1, n, self.C.hidden_size).to(self.C.device)

      for i in range(self.canvas_size):
        x_inp = samples[:, [i]]
        out, (C, c) = self.lstm(x_inp, (C, c))
        out = self.fc(out[:, 0, :])
        prob = th.sigmoid(out)
        sample_pixel = th.bernoulli(prob).unsqueeze(-1)  # n x 1 x 1
        if self.C.append_loc:
          loc = np.array([i // 28, i % 28]) / 27
          loc = th.FloatTensor(loc).to(self.C.device)
          loc = loc.view(1, 1, 2).repeat(n, 1, 1)
          sample_pixel = th.cat((sample_pixel, loc), dim=-1)
        samples = th.cat((samples, sample_pixel), dim=1)
      samples = samples[:,1:,0] if self.C.append_loc else samples[:,1:].squeeze(-1)
      samples = samples.view(n, *self.input_shape)
      return samples.cpu(), []
