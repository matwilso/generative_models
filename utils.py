from collections import defaultdict
import torchvision
import torch
import numpy as np

class AttrDict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__


def preproc(x):
    return (torch.as_tensor(x, dtype=torch.float32).transpose(1,-1) / 127.5) - 1.0

def unproc(img):
    img = (255 * (img.transpose(1,-1) + 1.0) / 2.0).detach().cpu().numpy().astype(np.uint8)
    img = img.reshape(-1, 32, 3)
    return img

# TODO: make this a total data handler. load, device, everything
class Dataset:
    pass

class CIFAR(Dataset):
    def __init__(self, H):
        self.H = H
        root = './data'
        self.train_data = torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=True)
        self.train_data.targets = np.array(self.train_data.targets)
        #self.test_data  = torchvision.datasets.CIFAR10(root, train=False, transform=None, target_transform=None, download=True)
        self.input_shape = [32, 32, 3]

    def sample_batch(self, bs, basic=False):
        N = self.train_data.data.shape[0]
        if basic:
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
        root = './data'
        self.train_data = torchvision.datasets.MNIST(root, train=True, download=True)
        self.input_shape = [28, 28, 1]

    def sample_batch(train_data, bs):
        import ipdb; ipdb.set_trace()
        N = train_data.data.shape[0]
        idxs = np.random.randint(0, N, bs)
        x = train_data.data[idxs]
        return preproc(x).to(self.H.device)
