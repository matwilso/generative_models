import random
from functools import partial
from pathlib import Path

import torch
from einops import rearrange, repeat
from torch.optim import Adam, lr_scheduler 

from gms import common
from gms.diffusion.gaussian_diffusion import GaussianDiffusion
from gms.diffusion.simple_unet import SimpleUnet

from torch.cuda import amp


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
    DG.sample_cond_w = -1.0
    DG.cf_drop_prob = 0.1
    DG.teacher_path = Path('.')
    DG.teacher_mode = 'step1'
    DG.lr_scheduler = 'none'
    DG.adam_ema = 0.999

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
            mean_type='v',
            num_steps=G.timesteps,
            sampler=G.sampler,
            teacher_net=self.teacher_net,
            teacher_mode=self.G.teacher_mode,
            sample_cond_w=G.sample_cond_w,
        )

        self.optimizer = Adam(self.net.parameters(), lr=G.lr, betas=(0.9, self.G.adam_ema), eps=1e-08, weight_decay=0)
        if G.pad32:
            self.size = 32
        else:
            self.size = 28
        self.scaler = amp.GradScaler()
        #self.scheduler = lr_scheduler.LinearLR

        self.scheduler = {
            'none': lr_scheduler.LambdaLR(self.optimizer, lambda x: 1),
            'linear': lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=G.epochs),
        }[self.G.lr_scheduler]

    def end_epoch(self):
        self.scheduler.step()

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
        return metrics

        #self.optimizer.zero_grad()
        ## sometimes drop out the class
        #y[torch.rand(y.shape[0]) < self.G.cf_drop_prob] = -1
        #loss, metrics = self.loss(x, y)
        #loss.backward()
        #self.optimizer.step()
        #return metrics

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
        # convert to a 5x5 grid for sample
        writer.add_image(
            'samples',
            rearrange(sample, '(n1 n2) c h w -> c (n1 h) (n2 w)', n1=5, n2=5),
            epoch,
        )
        # and for video as well
        def make_vid(name, arr):
            # TODO: make make_vid more used across other gen models
            vid = rearrange(arr, 't (n1 n2) c h w -> t c (n1 h) (n2 w)', n1=5, n2=5)[None]
            vid = repeat(vid, 'b t c h w -> b t (repeat c) h w', repeat=3)
            writer.add_video(
                name,
                vid,
                epoch,
                fps=self.G.timesteps / 3,
            )

        make_vid('sampling_process', zs)
        make_vid('x_preds', xs)
        make_vid('eps_preds', eps)

        torch.manual_seed(random.randint(0, 2**32))
