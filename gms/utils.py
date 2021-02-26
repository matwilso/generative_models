import inspect
import pathlib
from torchvision import transforms
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import torchvision
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import yaml
from torch import distributions as tdib

def args_type(default):
  if isinstance(default, bool):
    return lambda x: bool(['False', 'True'].index(x))
  if isinstance(default, int):
    return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
  if isinstance(default, pathlib.Path):
    return lambda x: pathlib.Path(x).expanduser()
  return type(default)

def count_vars(module):
  return sum([np.prod(p.shape) for p in module.parameters()])

class AttrDict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

class GM(torch.nn.Module):
  DC = AttrDict()  # default configuration. can be customized across models, both by changing the default value and by adding new options

  def __init__(self, C):
    super().__init__()
    self.C = C
    ## make the name be the name of the file (without .py)
    #self.name = pathlib.Path(inspect.getfile(self.__class__)).with_suffix('').name
    self.optimizer = None

  def train_step(self, x):
    """take one step on a batch to update the network"""
    assert hasattr(self, 'loss'), 'you are using the default train_step. this requires you to define a loss function that returns loss, metrics'
    if self.optimizer is None:
      self.optimizer = Adam(self.parameters(), self.C.lr)
    self.optimizer.zero_grad()
    loss, metrics = self.loss(x)
    loss.backward()
    self.optimizer.step()
    return metrics

  def evaluate(self, writer, x, epoch):
    assert False, "you need to implement the evaluate method. make some samples or something."

class Autoreg(GM):
  def evaluate(self, writer, x, epoch):
    samples, gen = self.sample(25)
    B, C, H, W = samples.shape
    samples = samples.reshape([B, C, H, W])
    writer.add_image('samples', combine_imgs(samples, 5, 5)[None], epoch)
    if len(gen) != 0:
      gen = torch.stack(gen).reshape([H*W, B, 1, H, W]).permute(1, 0, 2, 3, 4)
      writer.add_video('sampling_process', combine_imgs(gen, 5, 5)[None,:,None], epoch, fps=60)

class CategoricalHead(nn.Module):
  """take logits and produce a multinomial distribution independently"""
  def __init__(self, in_n, out_n, C):
    super().__init__()
    self.layer = nn.Linear(in_n, out_n)
  def forward(self, x, past_o=None):
    x = self.layer(x)
    return tdib.Multinomial(logits=x)

class BinaryHead(nn.Module):
  """take logits and produce a bernoulli distribution independently"""
  def __init__(self, in_n, out_n, C):
    super().__init__()
    self.layer = nn.Linear(in_n, out_n)
  def forward(self, x, past_o=None):
    x = self.layer(x)
    return tdib.Bernoulli(logits=x)

def dump_logger(logger, writer, i, C):
  print('=' * 30)
  print(i)
  for key in logger:
    val = np.mean(logger[key])
    writer.add_scalar(key, val, i)
    print(key, val)
  print(C.full_cmd)
  with open(pathlib.Path(C.logdir) / 'hps.yaml', 'w') as f:
    yaml.dump(C, f)
  print('=' * 30)
  writer.flush()
  return defaultdict(lambda: [])

def append_location(x):
  """add xy coords to every pixel"""
  XY = torch.stack(torch.meshgrid(torch.linspace(0, 1, 28), torch.linspace(0, 1, 28)), 0).to(x.device)
  return torch.cat([x, XY[None].repeat_interleave(x.shape[0], 0)], 1)

def combine_imgs(arr, row=5, col=5):
  """takes batch of video or image and pushes the batch dim into certain image shapes given by b,row,col"""
  if len(arr.shape) == 4:  # image
    BS, C, H, W = arr.shape
    assert BS == row * col and H == W == 28, (BS, row, col, H, W)
    x = arr.reshape([row, col, 28, 28]).permute(0, 2, 1, 3).flatten(0, 1).flatten(-2)
    return x
  elif len(arr.shape) == 5:  # video
    BS, T, C, H, W = arr.shape
    assert BS == row * col and H == W == 28, (BS, T, row, col, H, W)
    x = arr.reshape([row, col, T, 28, 28]).permute(2, 0, 3, 1, 4).flatten(1, 2).flatten(-2)
    return x
  else:
    raise NotImplementedError()

def load_mnist(bs, binarize=True):
  from torchvision import transforms
  from torchvision.datasets import MNIST
  import torch.utils.data as data

  tfs = [transforms.ToTensor()]
  if binarize:
    tfs += [lambda x: (x > 0.5).float()]
  else:
    tfs += [lambda x: x.float()]
  transform = transforms.Compose(tfs)
  train_dset = MNIST('data', transform=transform, train=True, download=True)
  test_dset = MNIST('data', transform=transform, train=False, download=True)

  train_loader = data.DataLoader(train_dset, batch_size=bs, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
  test_loader = data.DataLoader(test_dset, batch_size=bs, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
  return train_loader, test_loader