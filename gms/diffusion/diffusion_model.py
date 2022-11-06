import random
from functools import partial
from pathlib import Path

import torch
from einops import rearrange, repeat
from torch.optim import Adam

from gms import common
from gms.diffusion.gaussian_diffusion import GaussianDiffusion
from gms.diffusion.simple_unet import SimpleUnet


class DiffusionModel(common.GM):
    DG = common.AttrDict()  # default G
    DG.binarize = 0
    DG.timesteps = 250
    DG.hidden_size = 128
    DG.dropout = 0.0
    DG.sampler = 'ddim'
    DG.eval_heavy = 1
    # conditional models
    DG.class_cond = 1
    DG.cf_cond_w = 0.5
    DG.cf_drop_prob = 0.1
    DG.teacher_path = Path('.')

    def __init__(self, G):
        super().__init__(G)
        self.net = SimpleUnet(G)
        if self.G.teacher_path != Path('.'):
            self.teacher_ddim = torch.jit.load(self.G.teacher_path)
            # initialize student to teacher weights
            self.net.load_state_dict(self.teacher_ddim.net.state_dict())
        else:
            self.teacher_ddim = None

        self.diffusion = GaussianDiffusion(
            mean_type='v',
            num_steps=G.timesteps,
            sampler=G.sampler,
            teacher_ddim=self.teacher_ddim,
            cond_w=G.cf_cond_w,
        )

        self.optimizer = Adam(self.parameters(), lr=G.lr)
        if G.pad32:
            self.size = 32
        else:
            self.size = 28

    def forward(self, x, i, guide=None):
        return self.diffusion.ddim_step(net=partial(self.net, guide=guide), logsnr_t=i, logsnr_s=i, z_t=x)

    def save(self, path, test_x, test_y):
        # save both normal version and jitted version. jitted version is nicer for loading
        super().save(path, test_x)
        torch_i = torch.zeros(test_x.shape[0], device=test_x.device)
        model_path = path / 'model.jit.pt'
        self.forward(test_x, torch_i, test_y)
        jit_step = torch.jit.trace(self, (test_x, torch_i, test_y))
        torch.jit.save(jit_step, model_path)

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        # sometimes drop out the class
        y[torch.rand(y.shape[0]) < self.G.cf_drop_prob] = -1
        loss, metrics = self.loss(x, y)
        loss.backward()
        self.optimizer.step()
        return metrics

    def loss(self, x, y):
        metrics = self.diffusion.training_losses(net=partial(self.net, guide=y), x=x)
        metrics = {key: val.mean() for key, val in metrics.items()}
        loss = metrics['loss']
        return loss, metrics

    def sample(self, n, y=None):
        with torch.no_grad():
            noise = torch.randn((n, 1, self.size, self.size), device=self.G.device)
            samples = self.diffusion.sample(net=partial(self.net, guide=y), init_x=noise)
            return samples[-1]

    def evaluate(self, writer, x, y, epoch):
        # draw samples and visualize the sampling process
        def proc(x):
            x = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu()
            if self.G.pad32:
                x = x[..., 2:-2, 2:-2]
            return x

        # TODO: show unconditional samples as well

        torch.manual_seed(0)
        noise = torch.randn((25, 1, self.size, self.size), device=x.device)
        labels = torch.arange(25, dtype=torch.long, device=x.device) % 10
        preds = self.diffusion.sample(net=partial(self.net, guide=labels), init_x=noise)
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
            fps=self.G.timesteps / 3,
        )
        torch.manual_seed(random.randint(0, 2**32))
