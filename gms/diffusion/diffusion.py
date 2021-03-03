import torch as th
from torch import nn
from torch.optim import Adam
import gms
from gms import utils
from .basicnet import BasicNet, SiLU, timestep_embedding
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps

class DiffusionModel(utils.GM):
  DC = utils.AttrDict()  # default C
  DC.binarize = 0  # don't binarize the data
  DC.schedule = 'cosine'
  DC.loss_type = 'mse'

  def __init__(self, C):
    super().__init__(C)
    self.net = BasicNet()
    betas = gd.get_named_beta_schedule(C.schedule, 500)
    loss_type = {
      'mse': gd.LossType.MSE,
      'rmse': gd.LossType.RESCALED_MSE,
      'kl': gd.LossType.KL,
      'rkl': gd.LossType.RESCALED_KL,
    }
    self.diffusion = gd.GaussianDiffusion(betas=betas, loss_type=loss_type[C.loss_type])
    self.optimizer = Adam(self.parameters(), lr=C.lr)

  def train_step(self, y):
    x = 2 * y - 1
    self.optimizer.zero_grad()
    t = th.randint(0, 500, (x.shape[0],)).to(x.device)
    terms = self.diffusion.training_losses(self.net, x, t)
    loss = terms['loss'].mean()
    loss.backward()
    self.optimizer.step()
    return {'loss': loss}

  def evaluate(self, writer, x, epoch):
    sample = self.diffusion.p_sample_loop(
      self.net,
      (25, 1, 28, 28),
      clip_denoised=True
    )
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    writer.add_image('samples', utils.combine_imgs(sample, 5, 5)[None], epoch)

  def sample(self, x):
    pass
  
  def forward(self, x, emb):
    return self.net(x, emb)
