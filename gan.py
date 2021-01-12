import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
from itertools import chain, count
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F

# TODO: try out spec norm

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, stride=2)
        self.c2 = nn.Conv2d(32, 64, 3, stride=2)
        self.c3 = nn.Conv2d(64, 128, 3, stride=2)
        self.c4 = nn.Conv2d(128, 256, 3, stride=1)
        self.f1 = nn.Linear(256, 1)

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
        #x = torch.sigmoid(x)
        return x

class Generator(nn.Module):
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
        return x

if __name__ == '__main__':
    from utils import CIFAR, MNIST
    device = 'cuda'
    ds = CIFAR(device)
    disc = Discriminator().to(device)
    gen = Generator().to(device)

    # Setup Adam optimizers for both G and D
    disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
    gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.999))

    bs = 256

    #fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.

    for i in count():
        ## Train with all-real batch
        # Discriminator
        # real example
        disc_optim.zero_grad()
        batch = ds.sample_batch(bs).to(device)
        real_disc = disc(batch)
        # fake example
        noise = torch.randn(bs, 128).to(device)
        fake = gen(noise)
        fake_disc = disc(fake)
        # backward
        disc_loss = (real_disc - fake_disc).mean()
        disc_loss += 0.0001*(real_disc**2 + fake_disc**2).mean()
        #disc_loss = real_disc.mean() - fake_disc.mean()
        disc_loss.backward()
        disc_optim.step()

        # Generator
        gen_optim.zero_grad()
        fake = gen(noise)
        fake_disc = disc(fake)
        gen_loss = fake_disc.mean()
        gen_loss.backward()
        gen_optim.step()

        if i % 1000 == 0:
            pred = ds.export(fake[:10])
            truth = ds.export(batch[:10])
            img = np.concatenate([truth, np.zeros_like(truth), pred], axis=1)
            plt.imsave('test2.png', img)
            print(i, disc_loss.item())

