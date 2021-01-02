import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
from itertools import chain, count
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F

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
        x = torch.sigmoid(x)
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
    root = './'
    train_data = torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=True)
    #test_data  = torchvision.datasets.CIFAR10(root, train=False, transform=None, target_transform=None, download=True)
    #print(len(train_data), len(test_data))
    disc = Discriminator()
    gen = Generator()

    # Setup Adam optimizers for both G and D
    disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
    gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.999))
    imgs = (torch.as_tensor(train_data.data[:16], dtype=torch.float32).transpose(1,-1) / 127.5) - 1.0
    #imgs = imgs#.cuda()

    bce = nn.BCELoss()
    #fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.

    def export(img):
        img = (255 * (img.transpose(1,-1) + 1.0) / 2.0).detach().cpu().numpy().astype(np.uint8)
        img = img.reshape(-1, 32, 3)
        return img

    for i in count():
        ## Train with all-real batch
        # Discriminator
        # real example
        disc_optim.zero_grad()
        real_disc = disc(imgs)
        real_loss = bce(real_disc, torch.ones_like(real_disc))
        # fake example
        noise = torch.randn(16, 128)
        fake = gen(noise)
        fake_disc = disc(fake)
        fake_loss = bce(fake_disc, torch.zeros_like(fake_disc))
        # backward
        loss = real_loss + fake_loss
        loss.backward()
        disc_optim.step()

        # Generator
        gen_optim.zero_grad()
        fake = gen(noise)
        fake_disc = disc(fake)
        gen_loss = bce(fake_disc, torch.ones_like(fake_disc))
        gen_loss.backward()
        gen_optim.step()

        if i % 100 == 0:
            pred = export(fake)
            truth = export(imgs)
            img = np.concatenate([truth, np.zeros_like(truth), pred], axis=1)
            plt.imsave('test.png', img)
            print(real_loss.item(), fake_loss.item(), gen_loss.item())


