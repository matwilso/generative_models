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
H.lr = 1e-4
H.class_cond = 0
H.hidden_size = 512
H.append_loc = 1
H.overfit_batch = 0

# TODO: record bits/dim
# TODO: try interpolation


def append_location(x):
  """add xy coords to every pixel"""
  XY = torch.stack(torch.meshgrid(torch.linspace(0, 1, x.shape[-2]), torch.linspace(0, 1, x.shape[-1])), 0)
  return torch.cat([x, XY[None].repeat_interleave(x.shape[0], 0).to(x.device)], 1)


class MaskedLinear(nn.Linear):
  """ same as Linear except has a configurable mask on the weights """

  def __init__(self, in_features, out_features, bias=True):
    super().__init__(in_features, out_features, bias)
    self.register_buffer('mask', torch.ones(out_features, in_features))

  def set_mask(self, mask):
    self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

  def forward(self, input):
    return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
  def __init__(self, H):
    super().__init__()
    self.nin = 784
    self.nout = 784
    self.hidden_sizes = [H.hidden_size]*3
    self.H = H

    # define a simple MLP neural net
    self.net = []
    hs = [self.nin] + self.hidden_sizes + [self.nout]
    for h0, h1 in zip(hs, hs[1:]):
      self.net.extend([
          MaskedLinear(h0, h1),
          nn.ReLU(),
      ])
    self.net.pop()  # pop the last ReLU for the output layer
    self.net = nn.Sequential(*self.net)

    self.m = {}
    self.create_mask()  # builds the initial self.m connectivity

  def create_mask(self):
    L = len(self.hidden_sizes)
    # sample the order of the inputs and the connectivity of all neurons
    self.m[-1] = np.arange(self.nin)
    for l in range(L):
      self.m[l] = np.random.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])

    # you are flexible to use the neurons for whatever. like the data could have come from wherever.
    # you just need to assure that no information can propagate from anywhere earlier in the image.
    # the output that connects to pixel 0 can never see information from pixels 1-783.

    # construct the mask matrices
    # only activate connections where information comes from a lower numerical rank
    masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
    masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])
    #import ipdb; ipdb.set_trace()

    # set the masks in all MaskedLinear layers
    layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
    for l, m in zip(layers, masks):
      l.set_mask(m)

  def nll(self, x):
    x = x[0].to(self.H.device)
    x = x.view(-1, 784)  # Flatten image
    logits = self.net(x)
    return F.binary_cross_entropy_with_logits(logits, x)

  def sample(self, n):
    samples = torch.zeros(n, 784).to(self.H.device)
    # set the pixels 1 by 1 in raster order.
    # choose pixel 0, then based on that choose pixel 1, then based on both of those choose pixel 2. etc and so on.
    # This works ok, because it is used to this version of information propagation.
    # Normally, you can't see the future. And here you can't either. So the same condition is enforced.
    imgs = []
    with torch.no_grad():
      for i in range(784):
        logits = self.net(samples)[:, i]
        probs = torch.sigmoid(logits)
        samples[:, i] = torch.bernoulli(probs)
        x = samples.view(n, 1, 28,28).cpu()
        imgs += [x]
        #plt.imsave(f'gifs/{i}.png', x.numpy())
      samples = samples.view(n, 1, 28, 28)
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
  model = MADE(H).to(H.device)
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
