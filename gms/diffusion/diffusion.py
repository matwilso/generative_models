import torch as th
from torch import nn
from torch.optim import Adam
import gms
from gms import utils
from .basicnet import BasicNet, timestep_embedding
from .unet import UNetModel
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps

class DiffusionModel(utils.GM):
  DC = utils.AttrDict()  # default C
  DC.binarize = 0  # don't binarize the data
  DC.schedule = 'cosine'
  DC.loss_type = 'mse'
  DC.timesteps = 500
  DC.hidden_size = 128

  def __init__(self, C):
    super().__init__(C)
    self.net = UNetModel(1, 64, 2, 3, [4])
    #self.net = BasicNet(C)
    betas = gd.get_named_beta_schedule(C.schedule, C.timesteps)
    loss_type = {
        'mse': gd.LossType.MSE,
        'rmse': gd.LossType.RESCALED_MSE,
        'kl': gd.LossType.KL,
        'rkl': gd.LossType.RESCALED_KL,
    }
    self.diffusion = gd.GaussianDiffusion(betas=betas, loss_type=loss_type[C.loss_type])
    self.optimizer = Adam(self.parameters(), lr=C.lr)

  def train_step(self, x):
    self.optimizer.zero_grad()
    loss, metrics = self.loss(x)
    loss.backward()
    self.optimizer.step()
    return metrics

  def loss(self, x):
    t = th.randint(0, self.C.timesteps, (x.shape[0],)).to(x.device)
    metrics = self.diffusion.training_losses(self.net, x, t)
    metrics = {key: val.mean() for key, val in metrics.items()}
    loss = metrics['loss']
    return loss, metrics

  def evaluate(self, writer, x, epoch):
    sample = self.diffusion.p_sample_loop_progressive(
        self.net,
        (25, 1, 32, 32),
        clip_denoised=True
    )
    samples, preds = [], []
    def p(x): return ((x + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu()[...,2:-2,2:-2]
    for s in sample:
      samples += [p(s['sample'])]
      preds += [p(s['pred_xstart'])]

    sample = samples[-1]
    writer.add_image('samples', utils.combine_imgs(sample, 5, 5)[None], epoch)

    gs = th.stack(samples)
    gp = th.stack(preds)
    writer.add_video('sampling_process', utils.combine_imgs(gs.permute(1, 0, 2, 3, 4), 5, 5)[None, :, None], epoch, fps=60)
    writer.add_video('pred_process', utils.combine_imgs(gp.permute(1, 0, 2, 3, 4), 5, 5)[None, :, None], epoch, fps=60)

  def sample(self, x):
    pass

  def forward(self, x, emb):
    return self.net(x, emb)
