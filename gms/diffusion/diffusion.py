import torch as th
from torch import nn
from torch.optim import Adam
import gms
from gms import utils
from .simple_unet import SimpleUnet
from . import gaussian_diffusion as gd

class DiffusionModel(utils.GM):
  DC = utils.AttrDict()  # default C
  DC.binarize = 0
  DC.timesteps = 500 # seems to work pretty well for MNIST
  DC.hidden_size = 128
  DC.dropout = 0.0

  def __init__(self, C):
    super().__init__(C)
    self.net = SimpleUnet(C)
    self.diffusion = gd.GaussianDiffusion(C.timesteps)
    self.optimizer = Adam(self.parameters(), lr=C.lr)
    if C.pad32:
      self.size = 32
    else:
      self.size = 28

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
    # draw samples and visualize the sampling process
    def proc(x):
      x = ((x + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu()
      if self.C.pad32:
        x = x[...,2:-2,2:-2]
      return x
    all_samples = self.diffusion.p_sample( self.net, (25, 1, self.size, self.size))
    samples, preds = [], []
    for s in all_samples:
      samples += [proc(s['sample'])]
      preds += [proc(s['pred_xstart'])] 

    sample = samples[-1]
    writer.add_image('samples', utils.combine_imgs(sample, 5, 5)[None], epoch)

    gs = th.stack(samples)
    gp = th.stack(preds)
    writer.add_video('sampling_process', utils.combine_imgs(gs.permute(1, 0, 2, 3, 4), 5, 5)[None, :, None], epoch, fps=60)
    # at any point, we can guess at the true value of x_0 based on our current noise
    # this produces a nice visualization
    writer.add_video('pred_xstart', utils.combine_imgs(gp.permute(1, 0, 2, 3, 4), 5, 5)[None, :, None], epoch, fps=60)