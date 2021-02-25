import pathlib
from torchvision import transforms
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import torchvision
import torch
import numpy as np
import yaml

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
  DC = AttrDict()  # default configuration. can be customized across models

  def __init__(self):
    super().__init__()

  def train_step(self, batch):
    raise NotImplementedError()

  def train_step(self, batch):
    """take one step on a batch to update the network"""
    assert hasattr(self, 'loss'), 'you are using the default train_step. this requires you to define a loss function that returns loss, metrics'
    assert hasattr(self, 'optimizer'), 'you are using the default train_step. this requires you to use define self.optimizer'
    self.optimizer.zero_grad()
    loss, metrics = self.loss(batch)
    loss.backward()
    self.optimizer.step()
    return metrics

  def evaluate(self, batch):
    raise NotImplementedError()

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

def combine_imgs(arr, row=5, col=5):
  """takes batch of video or image and pushes the batch dim into certain image shapes given by b,row,col"""
  if len(arr.shape) == 4:  # image
    BS, C, H, W = arr.shape
    assert BS == row * col and H == W == 28, (BS, row, col, H, W)
    x = arr.reshape([row, col, 28, 28]).permute(0, 2, 1, 3).flatten(0, 1).flatten(-2)
    return x
  elif len(arr.shape) == 5:  # video
    BS, T, C, H, W = arr.shape
    assert BS == row * col and H == W == 28, (BS, row, col, H, W)
    x = arr.reshape([row, col, T, 28, 28]).permute(2, 0, 3, 1, 4).flatten(0, 1).flatten(-2)
  else:
    raise NotImplementedError()

def plot25(writer, name, arr, i):

  l = []
  for a in args:
    upr = torch.reshape(a, [-1, 28]).cpu().detach().numpy()
    l += [upr, np.zeros_like(upr)]
  img = np.concatenate(l, axis=-1)
  plt.imsave('test.png', img)
  writer.add_image(name, img[..., None], i, dataformats='HWC')

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
