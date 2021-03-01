import torch as th
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from gms import utils

# Based on DCGAN, but scaled down to 28x28 MNIST
# mostly taken from https://github.com/pytorch/examples/blob/6c8e2bab4d45f2386929c83bb4480c18d2b660fd/dcgan/main.py

class GAN(utils.GM):
  DC = utils.AttrDict()  # default C
  DC.noise_size = 128
  DC.binarize = 0 # don't binarize the data for GAN, because it's easier to deal with this way.
  DC.lr = 1e-4

  def __init__(self, C):
    super().__init__(C)
    self.disc = Discriminator(C)
    self.gen = Generator(C)
    self.disc_optim = Adam(self.disc.parameters(), lr=C.lr, betas=(0.5, 0.999))
    self.gen_optim = Adam(self.gen.parameters(), lr=C.lr, betas=(0.5, 0.999))
    self.bce = nn.BCELoss()
    self.fixed_noise = th.randn(25, C.noise_size).to(C.device)

  def train_step(self, x):
    bs = x.shape[0]
    noise = th.randn(bs, self.C.noise_size).to(self.C.device)
    # DISCRIMINATOR TRAINING - distinguish between real images and generator images
    self.disc_optim.zero_grad()
    # label real as 1 and learn to predict that
    true_output = self.disc(x)
    loss_real = self.bce(true_output, th.ones_like(true_output))
    loss_real.backward()
    # label fake as 0 and learn to predict that
    fake = self.gen(noise)
    fake_output = self.disc(fake.detach())
    loss_fake = self.bce(fake_output, th.zeros_like(fake_output))
    loss_fake.backward()
    self.disc_optim.step()
    # GENERATOR TRAINING - try to produce outputs discriminator thinks is real
    self.gen_optim.zero_grad()
    output = self.disc(fake)
    gen_loss = self.bce(output, th.ones_like(output)) 
    gen_loss.backward()
    self.gen_optim.step()
    metrics = {'disc/loss': loss_fake+loss_real, 'disc/loss_fake': loss_fake, 'disc/loss_real': loss_real, 'gen/loss': gen_loss}
    return metrics

  def sample(self, n):
    fake = self.gen(th.randn(n, self.C.noise_size).to(self.C.device))
    return fake

  def evaluate(self, writer, x, epoch):
    samples = self.sample(25)
    writer.add_image('samples', utils.combine_imgs(samples, 5, 5)[None], epoch)
    # fixed noise
    fixed_sample = self.gen(self.fixed_noise)
    writer.add_image('fixed_noise', utils.combine_imgs(fixed_sample, 5, 5)[None], epoch)

class Generator(nn.Module):
  def __init__(self, C):
    super().__init__()
    H = C.hidden_size
    self.net = nn.Sequential(
        nn.ConvTranspose2d(C.noise_size, H, 5, 1),
        nn.BatchNorm2d(H),
        nn.ReLU(),
        nn.ConvTranspose2d(H, H, 4, 2),
        nn.BatchNorm2d(H),
        nn.ReLU(),
        nn.ConvTranspose2d(H, H, 4, 2),
        nn.BatchNorm2d(H),
        nn.ReLU(),
        nn.ConvTranspose2d(H, 1, 3, 1),
        nn.Sigmoid(),)
    self.apply(weights_init)

  def forward(self, x):
    x = self.net(x[...,None,None])
    return x

class Discriminator(nn.Module):
  def __init__(self, C):
    super().__init__()
    H = C.hidden_size
    self.net = nn.Sequential(
        nn.Conv2d(1, H, 3, 2),
        nn.LeakyReLU(),
        nn.Conv2d(H, H, 3, 2),
        nn.BatchNorm2d(H),
        nn.LeakyReLU(),
        nn.Conv2d(H, H, 3, 1),
        nn.BatchNorm2d(H),
        nn.LeakyReLU(),
        nn.Conv2d(H, 1, 3, 2),
        nn.Flatten(-3),
        nn.Sigmoid()
    )
    self.apply(weights_init)

  def forward(self, x):
    return self.net(x)

# DCGAN initialization
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)