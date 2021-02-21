import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as tdib

class GaussHead(nn.Module):
  def forward(self, x):
    mu, log_std = x.chunk(2,-1)
    std = F.softplus(log_std)
    return tdib.Normal(mu, std)

class DownConv(nn.Module):
  def __init__(self, out_size, C):
    super().__init__()
    self.C = C
    H = self.C.hidden_size
    self.net = nn.Sequential(
        nn.Conv2d(1, H, 3, 2),
        nn.ReLU(),
        nn.Conv2d(H, H, 3, 2),
        nn.ReLU(),
        nn.Conv2d(H, H, 3, 1),
        nn.ReLU(),
        nn.Conv2d(H, 2 * self.C.z_size, 3, 2),
    )

  def forward(self, x):
    return self.net(x)

class UpConv(nn.Module):
  def __init__(self, in_size, C):
    super().__init__()
    self.C = C
    H = self.C.hidden_size
    self.net = nn.Sequential(
        nn.ConvTranspose2d(self.C.z_size, H, 5, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(H, H, 4, 2),
        nn.ReLU(),
        nn.ConvTranspose2d(H, H, 4, 2),
        nn.ReLU(),
        nn.ConvTranspose2d(H, 1, 3, 1),
    )

  def forward(self, x):
    x = self.net(x[..., None, None])
    return x