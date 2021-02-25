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
  DC = AttrDict() # default configuration. can be customized across models

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
  return defaultdict(lambda: [])

def plot_samples(name, writer, i, *args):
  l = []
  for a in args:
    upr = torch.reshape(a, [10 * 28, 28]).cpu().detach().numpy()
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
