import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
from itertools import chain, count
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F

# TODO: VQ VAE may be worth doing. but maybe as a separate repo.

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, stride=2)
        self.c2 = nn.Conv2d(32, 64, 3, stride=2)
        self.c3 = nn.Conv2d(64, 128, 3, stride=2)
        self.c4 = nn.Conv2d(128, 256, 3, stride=1)

    def forward(self, x):
        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.c4(x)
        x = x.flatten(1,3)
        mu, log_std = x.split(128, dim=1)
        std = F.softplus(log_std)
        norm = tdib.Normal(mu, std)
        prior = tdib.Normal(torch.zeros_like(mu), torch.ones_like(std))
        prior_loss = tdib.kl_divergence(norm, prior)
        return prior_loss.mean(), norm.rsample(), mu

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.d2 = nn.ConvTranspose2d(64, 32, 3, stride=2)
        self.d3 = nn.ConvTranspose2d(32, 16, 3, stride=2)
        self.d4 = nn.ConvTranspose2d(16, 3, 4, stride=2)

    def forward(self, x):
        x = self.d1(x[...,None,None])
        x = F.relu(x)
        x = self.d2(x)
        x = F.relu(x)
        x = self.d3(x)
        x = F.relu(x)
        x = self.d4(x)
        x = tdib.Normal(x, 1)
        return x

if __name__ == '__main__':
    from utils import preproc, export, sample_batch
    root = './data'
    train_data = torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=True)
    #test_data  = torchvision.datasets.CIFAR10(root, train=False, transform=None, target_transform=None, download=True)
    #print(len(train_data), len(test_data))
    device = 'cuda'
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    optimizer = Adam(chain(encoder.parameters(), decoder.parameters()), lr=3e-4)
    imgs = (torch.as_tensor(train_data.data[:16], dtype=torch.float32).transpose(1,-1) / 127.5) - 1.0
    bs = 1024

    for i in count():
        optimizer.zero_grad()
        batch = sample_batch(train_data, bs).to(device)
        prior_loss, code, mu = encoder(batch)
        recondist = decoder(code)

        recon_loss = -recondist.log_prob(batch).mean()

        loss = prior_loss + recon_loss
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            reconmu = decoder(mu[:10]).mean
            reconsamp = decoder(torch.randn(mu[:10].shape).to(device)).mean
            pred = export(reconmu)
            sample = export(reconsamp)
            truth = export(batch[:10])
            black = np.zeros_like(truth)
            img = np.concatenate([truth, black, pred, black, sample], axis=1)
            plt.imsave('test.png', img)
            print(i, loss.item(), prior_loss.item(), recon_loss.item())



