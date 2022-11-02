import random

import torch
from einops import rearrange, repeat
from torch.optim import Adam

from gms import common
from gms.diffusion.diffusion2.diffusion_handler import DiffusionHandler
from gms.diffusion.diffusion2.simple_unet import SimpleUnet


class VDiffusionModel(common.GM):
    DG = common.AttrDict()  # default G
    DG.binarize = 0
    DG.timesteps = 500  # seems to work pretty well for MNIST
    DG.hidden_size = 128
    DG.dropout = 0.0
    DG.sampler = 'ddim'

    def __init__(self, G):
        super().__init__(G)
        self.net = SimpleUnet(G)
        self.diffusion_handler = DiffusionHandler(
            mean_type='v', num_steps=G.timesteps, sampler=G.sampler
        )

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
        metrics = self.diffusion_handler.training_losses(net=self.net, x=x)
        metrics = {key: val.mean() for key, val in metrics.items()}
        loss = metrics['loss']
        return loss, metrics

    def sample(self, n):
        noise = torch.randn((n, 1, self.size, self.size), device=self.G.device)
        samples = self.diffusion_handler.sample(net=self.net, init_x=noise)
        return samples[-1]

    def evaluate(self, writer, x, epoch):
        # draw samples and visualize the sampling process
        def proc(x):
            x = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu()
            if self.G.pad32:
                x = x[..., 2:-2, 2:-2]
            return x

        torch.manual_seed(0)
        noise = torch.randn((25, 1, self.size, self.size), device=x.device)

        preds = self.diffusion_handler.sample(net=self.net, init_x=noise)
        preds = proc(preds)
        sample = preds[-1]
        # convert to a 5x5 grid for sample
        writer.add_image(
            'samples',
            rearrange(sample, '(n1 n2) c h w -> c (n1 h) (n2 w)', n1=5, n2=5),
            epoch,
        )
        # and for video as well
        vid = rearrange(preds, 't (n1 n2) c h w -> t c (n1 h) (n2 w)', n1=5, n2=5)[None]
        vid = repeat(vid, 'b t c h w -> b t (repeat c) h w', repeat=3)
        writer.add_video(
            'sampling_process',
            vid,
            epoch,
            fps=60,
        )
        torch.manual_seed(random.randint(0, 2**32))
