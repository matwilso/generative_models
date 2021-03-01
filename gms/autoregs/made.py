import numpy as np
from torch.optim import Adam
import torch as th
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from gms import utils

class MADE(utils.Autoreg):
  DC = utils.AttrDict()
  DC.hidden_size = 1024
  def __init__(self, C):
    super().__init__(C)
    self.nin = 784
    self.nout = 784
    self.hidden_sizes = [C.hidden_size]*3

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
    """
    you are flexible to use the neurons for whatever. like the data could have come from wherever.
    you just need to assure that no information can propagate from anywhere earlier in the image.
    the output that connects to pixel 0 can never see information from pixels 1-783.
    """
    L = len(self.hidden_sizes)
    # sample the order of the inputs and the connectivity of all neurons
    self.m[-1] = np.arange(self.nin)
    for l in range(L):
      self.m[l] = np.random.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])
    # construct the mask matrices
    # only activate connections where information comes from a lower numerical rank
    masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
    masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])
    # set the masks in all MaskedLinear layers
    layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
    for l, m in zip(layers, masks):
      l.set_mask(m)

  def loss(self, x):
    x = x.to(self.C.device)
    x = x.view(-1, 784)  # Flatten image
    logits = self.net(x)
    loss = -tdib.Bernoulli(logits=logits).log_prob(x).mean()
    return loss, {'nlogp': loss}

  def sample(self, n):
    samples = th.zeros(n, 784).to(self.C.device)
    # set the pixels 1 by 1 in raster order.
    # choose pixel 0, then based on that choose pixel 1, then based on both of those choose pixel 2. etc and so on.
    # This works ok, because it is used to this version of information propagation.
    # Normally, you can't see the future. And here you can't either. So the same condition is enforced.
    steps = []
    with th.no_grad():
      for i in range(784):
        logits = self.net(samples)[:, i]
        probs = th.sigmoid(logits)
        samples[:, i] = th.bernoulli(probs)
        steps += [samples.view(n, 1, 28,28).cpu()]
        #plt.imsave(f'gifs/{i}.png', x.numpy())
      samples = samples.view(n, 1, 28, 28)
    return samples.cpu(), steps

class MaskedLinear(nn.Linear):
  """ same as Linear except has a configurable mask on the weights """
  def __init__(self, in_features, out_features, bias=True):
    super().__init__(in_features, out_features, bias)
    self.register_buffer('mask', th.ones(out_features, in_features))

  def set_mask(self, mask):
    self.mask.data.copy_(th.from_numpy(mask.astype(np.uint8).T))

  def forward(self, input):
    return F.linear(input, self.mask * self.weight, self.bias)
