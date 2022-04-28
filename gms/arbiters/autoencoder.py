from turtle import forward
import torch as th
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from gms import common

class Autoencoder(common.GM):
  DC = common.AttrDict()  # default C
  DC.z_size = 128

  def __init__(self, C):
    super().__init__(C)
    self.encoder = Encoder(C.z_size, C)
    self.decoder = Decoder(C.z_size, C)
    self.optimizer = Adam(self.parameters(), lr=C.lr)

  def forward(self, x):
    return self.encoder(x)

  def loss(self, x, y=None):
    z = self.encoder(x)
    decoded = self.decoder(z)
    recon_loss = -tdib.Bernoulli(logits=decoded).log_prob(x).mean((1, 2, 3))
    loss = (recon_loss).mean()
    metrics = {'vae_loss': loss, 'recon_loss': recon_loss.mean()}
    return loss, metrics

  def evaluate(self, writer, x, y, epoch, arbiter=None, classifier=None):
    """run samples and other evaluations"""
    z = self.encoder(x[:8])
    recon = self._decode(z)
    recon = th.cat([x[:8].cpu(), recon], 0)
    writer.add_image('reconstruction', common.combine_imgs(recon, 2, 8)[None], epoch)

  def _decode(self, x):
    return 1.0 * (th.sigmoid(self.decoder(x)) > 0.5).cpu()

class Encoder(nn.Module):
  def __init__(self, out_size, C):
    super().__init__()
    H = C.hidden_size
    self.net = nn.Sequential(
        nn.Conv2d(1, H, 3, 2),
        nn.ReLU(),
        nn.Conv2d(H, H, 3, 2),
        nn.ReLU(),
        nn.Conv2d(H, H, 3, 1),
        nn.ReLU(),
        nn.Conv2d(H, out_size, 3, 2),
        nn.Flatten(1, 3),
    )

  def forward(self, x):
    return self.net(x)

class Decoder(nn.Module):
  def __init__(self, in_size, C):
    super().__init__()
    H = C.hidden_size
    self.net = nn.Sequential(
        nn.ConvTranspose2d(in_size, H, 5, 1),
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
