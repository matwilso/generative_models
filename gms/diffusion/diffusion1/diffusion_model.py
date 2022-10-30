import random

import torch
from torch.optim import Adam

from gms import common
from gms.diffusion.diffusion1 import gaussian_diffusion as gd
from gms.diffusion.diffusion1.simple_unet import SimpleUnet


class DiffusionModel(common.GM):
    DG = common.AttrDict()  # default G
    DG.binarize = 0
    DG.timesteps = 500  # seems to work pretty well for MNIST
    DG.hidden_size = 128
    DG.dropout = 0.0

    def __init__(self, G):
        super().__init__(G)
        self.net = SimpleUnet(G)
        self.diffusion = gd.GaussianDiffusion(G.timesteps)
        self.optimizer = Adam(self.parameters(), lr=G.lr)
        if G.pad32:
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
        t = torch.randint(0, self.G.timesteps, (x.shape[0],)).to(x.device)
        metrics = self.diffusion.training_losses(self.net, x, t)
        metrics = {key: val.mean() for key, val in metrics.items()}
        loss = metrics['loss']
        return loss, metrics

    def sample(self, n):
        noise = torch.randn((n, 1, self.size, self.size), device=self.G.device)
        samples = self.diffusion.p_sample(
            self.net, (n, 1, self.size, self.size), noise=noise
        )
        return samples[-1]['sample']

    def evaluate(self, writer, x, epoch):
        # draw samples and visualize the sampling process
        def proc(x):
            x = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu()
            if self.G.pad32:
                x = x[..., 2:-2, 2:-2]
            return x

        torch.manual_seed(0)
        noise = torch.randn((25, 1, self.size, self.size), device=x.device)
        all_samples = self.diffusion.p_sample(
            self.net, (25, 1, self.size, self.size), noise=noise
        )
        samples, preds = [], []
        for s in all_samples:
            samples += [proc(s['sample'])]
            preds += [proc(s['pred_xstart'])]

        sample = samples[-1]
        writer.add_image('samples', common.combine_imgs(sample, 5, 5)[None], epoch)

        gs = torch.stack(samples)
        gp = torch.stack(preds)
        writer.add_video(
            'sampling_process',
            common.combine_imgs(gs.permute(1, 0, 2, 3, 4), 5, 5)[None, :, None],
            epoch,
            fps=60,
        )
        # at any point, we can guess at the true value of x_0 based on our current noise
        # this produces a nice visualization
        writer.add_video(
            'pred_xstart',
            common.combine_imgs(gp.permute(1, 0, 2, 3, 4), 5, 5)[None, :, None],
            epoch,
            fps=60,
        )
        torch.manual_seed(random.randint(0, 2**32))
