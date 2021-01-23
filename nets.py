import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as tdib

def conv(inp, out, ks=3, stride=2, bn=False, activ=nn.Identity()):
    return [nn.Conv2d(inp, out, ks, stride, bias=not bn),
            nn.BatchNorm2d(out) if bn else nn.Identity(),
            activ]

def deconv(inp, out, ks=3, stride=2, bn=False, activ=nn.Identity()):
    return [nn.ConvTranspose2d(inp, out, ks, stride, bias=not bn),
            nn.BatchNorm2d(out) if bn else nn.Identity(),
            activ]

L = nn.ModuleList
class E1(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.net = nn.Sequential(
            *conv(3, 512, bn=H.bn, activ=nn.ReLU()),
            *conv(512, 512, bn=H.bn, activ=nn.ReLU()),
            *conv(512, 512, bn=H.bn, activ=nn.ReLU()),
            *conv(512, 2*self.H.z_size, stride=1),
                )

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1,3)
        mu, log_std = x.split(self.H.z_size, dim=1)
        std = F.softplus(log_std)
        norm = tdib.Normal(mu, std)
        prior = tdib.Normal(torch.zeros_like(mu), torch.ones_like(std))
        prior_loss = tdib.kl_divergence(norm, prior)
        return prior_loss.mean(-1), norm.rsample(), mu

class D1(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.net = nn.Sequential(
            *deconv(self.H.z_size + 10*self.H.class_cond, 512, bn=H.bn, activ=nn.ReLU()),
            *deconv(512, 512, bn=H.bn, activ=nn.ReLU()),
            *deconv(512, 64, bn=H.bn, activ=nn.ReLU()),
            *deconv(64, 3, ks=4),
            )

    def forward(self, x, cc=None):
        if cc is not None:
            x = torch.cat([x, cc], -1)
        x = self.net(x[...,None,None])
        x = tdib.Normal(x, 1)
        return x



