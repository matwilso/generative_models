import random
from functools import partial
from pathlib import Path

import torch
from torch.cuda import amp
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
    DG.mean_type = 'v'
    DG.eval_heavy = 1
    # conditional models
    DG.class_cond = 1
    DG.sample_cond_w = -1.0
    DG.cf_drop_prob = 0.1
    DG.teacher_path = Path('.')
    DG.teacher_mode = 'step1'
    DG.lr_scheduler = 'none'

    def __init__(self, G):
        super().__init__(G)
        self.net = SimpleUnet(G)
        if self.G.teacher_path != Path('.') and self.G.weights_from == Path('.'):
            print("Loading teacher model")
            # initialize student to teacher weights
            self.load_state_dict(torch.load(self.G.teacher_path), strict=False)
            # make teacher itself and freeze it
            self.teacher_net = SimpleUnet(G)
            self.teacher_net.load_state_dict(self.net.state_dict().copy())
            self.teacher_net.eval()
            for param in self.teacher_net.parameters():
                param.requires_grad = False
        else:
            self.teacher_net = None

        self.diffusion = GaussianDiffusion(
            mean_type=G.mean_type,
            num_steps=G.timesteps,
            sampler=G.sampler,
            teacher_net=self.teacher_net,
            teacher_mode=self.G.teacher_mode,
            sample_cond_w=G.sample_cond_w,
        )

        self.optimizer = Adam(self.net.parameters(), lr=G.lr)
        if G.pad32:
            self.size = 32
        else:
            self.size = 28
        self.scaler = amp.GradScaler()

    def train_step(self, x, y):
        # train network using torch AMP for float16
        self.optimizer.zero_grad()
        # sometimes drop out the class
        y[torch.rand(y.shape[0]) < self.G.cf_drop_prob] = -1
        with amp.autocast():
            loss, metrics = self.loss(x, y)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        metrics['loss_scale'] = torch.tensor(self.scaler.get_scale())
        return metrics

    def loss(self, x, y):
        metrics = self.diffusion.training_losses(net=partial(self.net, guide=y), x=x)
        metrics = {key: val.mean() for key, val in metrics.items()}
        loss = metrics['loss']
        return loss, metrics

    def sample(self, n, y=None):
        with torch.no_grad():
            noise = torch.randn((n, 1, self.size, self.size), device=self.G.device)
            net = partial(self.net, guide=y)
            samples = self.diffusion.sample(net=net, init_x=noise, cond_w=0.5)[0]
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
        zs, xs, eps = self.diffusion.sample(
            net=partial(self.net, guide=labels), init_x=noise
        )
        zs, xs, eps = proc(zs), proc(xs), proc(eps)
        sample = zs[-1]
        common.write_grid(writer, 'samples', sample, epoch)
        common.write_gridvid(writer, 'sampling_process', zs, epoch)
        common.write_gridvid(writer, 'diffusion_model/eps', eps, epoch)
        common.write_gridvid(writer, 'diffusion_model/x', xs, epoch)
        torch.manual_seed(random.randint(0, 2**32))
