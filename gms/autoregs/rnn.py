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
H.lr = 1e-4
H.class_cond = 0
H.hidden_size = 512
H.append_loc = 1
H.overfit_batch = 0

def append_location(x):
  """add xy coords to every pixel"""
  XY = torch.stack(torch.meshgrid(torch.linspace(0, 1, x.shape[-2]), torch.linspace(0, 1, x.shape[-1])), 0)
  return torch.cat([x, XY[None].repeat_interleave(x.shape[0], 0).to(x.device)], 1)


class RNN(nn.Module):
  def __init__(self, H):
    super().__init__()
    self.H = H
    self.input_shape = input_shape = (1, 28, 28)
    self.input_channels = input_shape[0] + 2 if H.append_loc else input_shape[0]
    self.canvas_size = input_shape[1] * input_shape[2]
    self.lstm = nn.LSTM(self.input_channels, self.H.hidden_size, num_layers=1, batch_first=True)
    self.fc = nn.Linear(self.H.hidden_size, input_shape[0])

  def nll(self, x):
    x = x[0].to(self.H.device)
    batch_size = x.shape[0]
    x_inp = append_location(x) if self.H.append_loc else x

    # make LSTM operate over 1 pixel at a time.
    x_inp = x_inp.permute(0, 2, 3, 1).contiguous().view(batch_size, self.canvas_size, self.input_channels)
    # align it so we are predicting the next pixel. start with dummy first and feed everything put last real pixel.
    x_inp = torch.cat((torch.zeros(batch_size, 1, self.input_channels).to(self.H.device), x_inp[:, :-1]), dim=1)

    h0 = torch.zeros(1, x_inp.size(0), self.H.hidden_size).to(self.H.device)
    c0 = torch.zeros(1, x_inp.size(0), self.H.hidden_size).to(self.H.device)

    # Forward propagate LSTM
    out, _ = self.lstm(x_inp, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

    # Decode the hidden state of the last time step
    out = self.fc(out).squeeze(-1)  # b x 784

    return F.binary_cross_entropy_with_logits(out, x.view(batch_size, -1))

  def sample(self, n):
    with torch.no_grad():
      samples = torch.zeros(n, 1, self.input_channels).to(self.H.device)
      h = torch.zeros(1, n, self.H.hidden_size).to(self.H.device)
      c = torch.zeros(1, n, self.H.hidden_size).to(self.H.device)

      for i in range(self.canvas_size):
        x_inp = samples[:, [i]]
        out, (h, c) = self.lstm(x_inp, (h, c))
        out = self.fc(out[:, 0, :])
        prob = torch.sigmoid(out)
        sample_pixel = torch.bernoulli(prob).unsqueeze(-1)  # n x 1 x 1
        if self.H.append_loc:
          loc = np.array([i // 28, i % 28]) / 27
          loc = torch.FloatTensor(loc).to(self.H.device)
          loc = loc.view(1, 1, 2).repeat(n, 1, 1)
          sample_pixel = torch.cat((sample_pixel, loc), dim=-1)
        samples = torch.cat((samples, sample_pixel), dim=1)

      if self.H.append_loc:
        samples = samples[:, 1:, 0]  # only get sampled pixels, ignore location
      else:
        samples = samples[:, 1:].squeeze(-1)  # n x 784
      samples = samples.view(n, *self.input_shape)
      return samples.cpu()


if __name__ == '__main__':
  # TODO: use low beta 0.1
  # TODO: make network bigger
  from gms.utils import load_mnist
  H = utils.parseC(H)
  writer = SummaryWriter(H.logdir)
  logger = utils.dump_logger({}, writer, 0, H)
  train_ds, test_ds = load_mnist(H.bs)
  _batch = next(iter(train_ds))
  _batch[0] = _batch[0].to(H.device)
  model = RNN(H).to(H.device)
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
          loss = model.nll(batch)
          total_loss += loss * batch[0].shape[0]
        avg_loss = total_loss / len(test_ds.dataset)
      logger['test/bits_per_dim'] = avg_loss.item() / np.log(2)
    samples = model.sample(10)
    utils.plot_samples(writer, i, batch[0][:10], samples)
    writer.flush()
    model.train()

  for i in count():
    train_epoch()
    eval()
    logger = utils.dump_logger(logger, writer, i, H)

    if i >= H.done_n:
      break