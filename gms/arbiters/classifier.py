from turtle import forward
import torch as th
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from gms import common
from .autoencoder import Encoder

class Classifier(common.GM):

  def __init__(self, C):
    super().__init__(C)
    self.net = Encoder(10, C)
    self.optimizer = Adam(self.parameters(), lr=C.lr)

  def forward(self, x):
    return self.net(x)

  def loss(self, x, y):
    z = self.net(x)
    loss = F.cross_entropy(z, y)
    metrics = {'cross_entropy_loss': loss}
    return loss, metrics

  def evaluate(self, writer, x, y, epoch, arbiter=None, classifier=None):
    """run samples and other evaluations"""
    preds = self.net(x[:8]).argmax(1)
    imgs = x[:8].cpu()
    mask = preds == y[:8]
    imgs[mask] = 1 - imgs[mask]
    writer.add_image('reconstruction', common.combine_imgs(imgs, 1, 8)[None], epoch)