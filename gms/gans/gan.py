import torch
from torch import nn
from torch.optim import Adam

from gms import common

# Based on DCGAN, but scaled down to 28x28 MNIST
# mostly taken from https://github.com/pytorch/examples/blob/6c8e2bab4d45f2386929c83bb4480c18d2b660fd/dcgan/main.py


class GAN(common.GM):
    DG = common.AttrDict()  # default G
    DG.noise_size = 128
    # don't binarize the data for GAN, because it's easier to deal with this way.
    DG.binarize = 0
    DG.lr = 1e-4

    def __init__(self, G):
        super().__init__(G)
        self.disc = Discriminator(G)
        self.gen = Generator(G)
        #self.disc_optim = Adam(self.disc.parameters(), lr=G.lr, betas=(0.5, 0.999))
        #self.gen_optim = Adam(self.gen.parameters(), lr=G.lr, betas=(0.5, 0.999))
        self.disc_optim = Adam(self.disc.parameters(), lr=1e-4)
        self.gen_optim = Adam(self.gen.parameters(), lr=1e-4)
        self.bce = nn.BCELoss()
        self.fixed_noise = torch.randn(25, G.noise_size).to(G.device)

    def train_step(self, x, y=None):
        bs = x.shape[0]
        noise = torch.randn(bs, self.G.noise_size).to(self.G.device)
        # DISCRIMINATOR TRAINING - distinguish between real images and generator images
        self.disc_optim.zero_grad()
        # label real as 1 and learn to predict that
        true_output = self.disc(x)
        loss_real = self.bce(true_output, torch.ones_like(true_output))
        loss_real.backward()
        # label fake as 0 and learn to predict that
        fake = self.gen(noise)
        fake_output = self.disc(fake.detach())
        loss_fake = self.bce(fake_output, torch.zeros_like(fake_output))
        loss_fake.backward()
        self.disc_optim.step()
        # GENERATOR TRAINING - try to produce outputs discriminator thinks is real
        self.gen_optim.zero_grad()
        output = self.disc(fake)
        gen_loss = self.bce(output, torch.ones_like(output))
        gen_loss.backward()
        self.gen_optim.step()
        metrics = {
            'disc/loss': loss_fake + loss_real,
            'disc/loss_fake': loss_fake,
            'disc/loss_real': loss_real,
            'gen/loss': gen_loss,
        }
        return metrics

    def sample(self, n):
        fake = self.gen(torch.randn(n, self.G.noise_size).to(self.G.device))
        return fake

    def evaluate(self, writer, x, y, epoch):
        samples = self.sample(25)
        common.write_grid(writer, 'samples', samples, epoch)
        # fixed noise
        fixed_sample = self.gen(self.fixed_noise)
        common.write_grid(writer, 'fixed_noise', fixed_sample, epoch)


class Generator(nn.Module):
    def __init__(self, G):
        super().__init__()
        H = G.hidden_size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(G.noise_size, H, 5, 1),
            nn.GroupNorm(32, H),
            nn.ReLU(),
            nn.ConvTranspose2d(H, H, 4, 2),
            nn.GroupNorm(32, H),
            nn.ReLU(),
            nn.ConvTranspose2d(H, H, 4, 2),
            nn.GroupNorm(32, H),
            nn.ReLU(),
            nn.ConvTranspose2d(H, 1, 3, 1),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x):
        x = self.net(x[..., None, None])
        return x


class Discriminator(nn.Module):
    def __init__(self, G):
        super().__init__()
        H = G.hidden_size
        self.net = nn.Sequential(
            nn.Conv2d(1, H, 3, 2),
            nn.LeakyReLU(),
            nn.Conv2d(H, H, 3, 2),
            nn.GroupNorm(32, H),
            nn.LeakyReLU(),
            nn.Conv2d(H, H, 3, 1),
            nn.GroupNorm(32, H),
            nn.LeakyReLU(),
            nn.Conv2d(H, 1, 3, 2),
            nn.Flatten(-3),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)


# DCGAN initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('GroupNorm') != -1:
    #elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
