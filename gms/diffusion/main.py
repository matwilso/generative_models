import ipdb; ipdb.set_trace()
import torch as th
from torch import nn
import gms
from gms import utils
from gms.diffusion.basicnet import BasicNet, SiLU, timestep_embedding
from gms.diffusion.gaussian_diffusion import GaussianDiffusion

class DiffusionModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = BasicNet()
    time_embed_dim = model_channels * 4
    self.time_embed = nn.Sequential(
        nn.Linear(model_channels, time_embed_dim),
        SiLU(),
        nn.Linear(time_embed_dim, time_embed_dim),
    )

  def train_step(self, x):
    t = th.randint(0, 500, (32,)).float()
    emb = self.time_embed(timestep_embedding(t, self.model_channels))
    self.forward(x, emb)

  def sample(self, x):
    pass
  
  def forward(x, emb):
    return self.net(x, emb)

model = DiffusionModel()
train_ds, test_ds = utils.load_mnist(32)
batch = next(iter(train_ds))
model.train_step(batch[0])
