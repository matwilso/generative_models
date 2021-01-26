import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import torchvision
import torch
import numpy as np

def parseH(H):
  parser = argparse.ArgumentParser()
  for key, value in H.items():
    parser.add_argument(f'--{key}', type=type(value), default=value)
  H = parser.parse_args()
  return H


class AttrDict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__


def preproc(x):
  return (x / 127.5) - 1.0


#def unproc(img):
#  img = (255 * (img.transpose(1, -1) + 1.0) / 2.0).detach().cpu().numpy().astype(np.uint8)
#  img = img.reshape(-1, 32, 3)
#  return img

def unproc(img):
  img = img.transpose(1, -1).detach().cpu().numpy().astype(np.uint8)
  img = img.reshape(-1, 28, 1)
  return img


def dump_logger(logger, writer, i, H):
  print('=' * 30)
  print(i)
  for key in logger:
    val = np.mean(logger[key])
    writer.add_scalar(key, val, i)
    print(key, val)
  print(H.full_cmd)
  print('=' * 30)
  return defaultdict(lambda: [])


def plot_samples(writer, i, *args):
  l = []
  for a in args:
    upr = torch.reshape(a, [10*28, 28]).cpu().detach().numpy()
    #upr = unproc(a)
    #upr = a.detach().cpu().transpose(1, -1).numpy()
    l += [upr, np.zeros_like(upr)]
  img = np.concatenate(l, axis=-1)
  plt.imsave('test.png', img)
  writer.add_image('test', img[...,None], i, dataformats='HWC')

# TODO: make this a total data handler. load, device, everything

class Dataset:
  pass

class CIFAR(Dataset):
  def __init__(self, H):
    self.H = H
    root = '../data'
    self.train_data = torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=True)
    self.train_data.targets = np.array(self.train_data.targets)
    #self.test_data  = torchvision.datasets.CIFAR10(root, train=False, transform=None, target_transform=None, download=True)

  def sample_batch(self, bs, overfit_batch=False):
    N = self.train_data.data.shape[0]
    if overfit_batch:
      image = self.train_data.data[:bs]
      label = self.train_data.targets[:bs]
      return {'image': preproc(image).to(self.H.device), 'label': torch.as_tensor(x, dtype=torch.long).to(self.H.device)}
    else:
      idxs = np.random.randint(0, N, bs)
      image = self.train_data.data[idxs]
      label = self.train_data.targets[idxs]
      return {'image': preproc(image).to(self.H.device), 'label': torch.as_tensor(label, dtype=torch.long).to(self.H.device)}

class MNIST(Dataset):
  def __init__(self, H):
    self.H = H
    root = '../data'
    self.train_data = torchvision.datasets.MNIST(root, train=True, download=True)

  def sample_batch(self, bs, overfit_batch=False):
    N = self.train_data.data.shape[0]
    if overfit_batch:
      image = self.train_data.data[:bs]
      label = self.train_data.targets[:bs]
      return {'image': (image[:,None] / 255.0).to(self.H.device), 'label': label.to(self.H.device)}
    else:
      idxs = np.random.randint(0, N, bs)
      image = self.train_data.data[idxs]
      label = self.train_data.targets[idxs]
      return {'image': (image[:,None] / 255.0).to(self.H.device), 'label': label.to(self.H.device)}

