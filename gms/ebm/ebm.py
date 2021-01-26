import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
from itertools import chain, count
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, size):
        self.buf = np.zeros([size, 3, 32, 32])
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, img):
        self.buf[self.ptr] = img
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def store_n(self, img):
        shape = img.shape[0]
        end_idx = self.ptr + shape
        if end_idx <= self.max_size:  # normal operation
            self.buf[self.ptr:end_idx] = img
            self.ptr = (self.ptr + shape) % self.max_size
        else:  # handle wrap around
            overflow = (end_idx - self.max_size)
            top_off = shape - overflow
            self.buf[self.ptr:self.ptr + top_off] = img[:top_off]  # top off the last end of the array
            self.buf[:overflow] = img[top_off:]  # start over at beginning
            self.ptr = overflow
        self.size = min(self.size + shape, self.max_size)

    def sample_batch(self, batch_size, random=False):
        x = np.random.uniform(-1, 1, size=[batch_size, 3, 32, 32])
        if self.size < 100 or random:
            return x
        else:
            mask_idxs = np.nonzero(np.random.binomial(1, 0.95, size=batch_size))[0]
            buf_idxs = np.random.randint(0, self.size, size=len(mask_idxs))
            x[mask_idxs] = self.buf[buf_idxs]
            return torch.as_tensor(x, dtype=torch.float32)

class EBM(nn.Module):
    def __init__(self):
        super().__init__()
        spec = lambda x: torch.nn.utils.spectral_norm(x)
        self.c1 = spec(nn.Conv2d(3, 32, 3, stride=2))
        self.c2 = spec(nn.Conv2d(32, 64, 3, stride=2))
        self.c3 = spec(nn.Conv2d(64, 128, 3, stride=2))
        self.c4 = spec(nn.Conv2d(128, 256, 3, stride=1))
        self.f1 = spec(nn.Linear(256, 1))

    def forward(self, x):
        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.c4(x)
        x = F.relu(x)
        x = x.flatten(1,3)
        x = self.f1(x)
        return x

def langevin(ebm, x):
    x = torch.as_tensor(x, dtype=torch.float32).cuda()
    x.requires_grad_()
    ox = x.data.clone().detach()

    for i in range(60):
        ebm.zero_grad()
        x.data.add_(tdib.Normal(torch.zeros(x.shape), 0.005*torch.ones(x.shape)).sample().cuda())
        energy = ebm(x).sum()
        energy.backward()
        #print(f'{i} energy {energy}')
        x.grad.clamp_(-0.01, 0.01)
        x.data.add_(-10*x.grad)
        x.data.clamp_(-1,1)

    #print('delta g', ((g-og)**2).mean())
    #print('delta z', ((z-oz)**2).mean())
    return x.detach()

if __name__ == '__main__':
    root = './'
    train_data = torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=True)
    #test_data  = torchvision.datasets.CIFAR10(root, train=False, transform=None, target_transform=None, download=True)
    #print(len(train_data), len(test_data))
    rb = ReplayBuffer(int(1e4))
    ebm = EBM().cuda()
    #ebm = torch.nn.utils.spectral_norm(ebm)
    ebm_optim = Adam(ebm.parameters(), lr=1e-4, betas=(0.0, 0.999))
    imgs = (torch.as_tensor(train_data.data[:16], dtype=torch.float32).transpose(1,-1) / 127.5) - 1.0
    imgs = imgs.cuda()
    def export(img):
        img = (255 * (img.transpose(1,-1) + 1.0) / 2.0).detach().cpu().numpy().astype(np.uint8)
        img = img.reshape(-1, 32, 3)
        return img

    for i in count():
        # get data. sample positive. generate negative
        positive = imgs
        negative = rb.sample_batch(16)
        negative = langevin(ebm, negative)
        rb.store_n(negative.cpu())
        # EBM
        ebm_optim.zero_grad()
        E_xp = ebm(positive)
        E_xn = ebm(negative)
        ebm_loss = E_xp - E_xn
        l2_reg_loss = 1.0*(E_xp**2 + E_xn**2)
        loss = (ebm_loss + l2_reg_loss).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(ebm.parameters(), 100).item()
        ebm_optim.step()
        # TODO: use spectral norm
        # TODO: better network

        if i % 100 == 0:
            pred = export(negative.cpu())
            truth = export(imgs)
            img = np.concatenate([truth, np.zeros_like(truth), pred], axis=1)
            plt.imsave('test.png', img)
            print(i, loss.item(), ebm_loss.mean().item(), l2_reg_loss.mean().item(), grad_norm)
