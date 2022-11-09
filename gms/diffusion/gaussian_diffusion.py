from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from gms.diffusion.diffusion_utils import (
    broadcast_from_left,
    diffusion_forward,
    diffusion_reverse,
    get_logsnr_schedule,
    mean_flat,
    normal_kl,
    predict_eps_from_x,
    predict_v_from_x_and_eps,
    predict_x_from_eps,
    predict_x_from_v,
)


class GaussianDiffusion:
    def __init__(
        self,
        *,
        mean_type,
        num_steps,
        teacher_ddim=None,
        teacher_mode='step1',
        sampler='ddim',
        cf_cond_w=None,
    ):
        self.mean_type = mean_type
        self.num_steps = num_steps
        self.teacher_ddim = teacher_ddim
        self.logsnr_schedule_fn = get_logsnr_schedule(
            'cosine', logsnr_min=-20.0, logsnr_max=20.0
        )
        self.sampler = sampler
        self.cf_cond_w = cf_cond_w
        self.loss_weight_type = 'snr_trunc'
        if self.teacher_ddim is not None:
            assert teacher_mode in ['step1', 'step2']
            self.teacher_mode = teacher_mode
            if self.teacher_mode == 'step1':
                self.loss_weight_type = 'snr'

    def _run_model(self, *, net, z, logsnr):
        """
        f(net, z, logsnr) --> (x, eps, v)
        """
        # ok so just to diagram. we have several different ways to predict a value to train on
        # we can specify which one we want to predict.
        # we can predict the noise, the x itself, the v (direction to more noise in spherical interpolation),
        # or both x and eps and we weight between them based on the logsnr.

        # if we are not predicting x, we need to get it
        model_output = net(z, logsnr)
        if self.mean_type == 'eps':
            model_eps = model_output
            model_x = predict_x_from_eps(z=z, eps=model_eps, logsnr=logsnr)
        elif self.mean_type == 'x':
            model_x = model_output
        elif self.mean_type == 'v':
            model_v = model_output
            model_x = predict_x_from_v(z=z, v=model_v, logsnr=logsnr)
        elif self.mean_type == 'both':
            _model_x, _model_eps = torch.split(model_output, 2, dim=-1)
            # reconcile the two predictions
            model_x_eps = predict_x_from_eps(z=z, eps=_model_eps, logsnr=logsnr)
            wx = broadcast_from_left(torch.sigmoid(-logsnr), z.shape)
            model_x = wx * _model_x + (1.0 - wx) * model_x_eps
        else:
            raise NotImplementedError(self.mean_type)

        model_x = torch.clip(model_x, -1.0, 1.0)
        # even if we already computed these, we need to recompute since we clipped model_x
        # so we may have to compute x from eps, then clip x and compute eps from x.
        model_eps = predict_eps_from_x(z=z, x=model_x, logsnr=logsnr)
        model_v = predict_v_from_x_and_eps(x=model_x, eps=model_eps, logsnr=logsnr)

        return {'model_x': model_x, 'model_eps': model_eps, 'model_v': model_v}

    def predict(
        self,
        *,
        net,
        z_t,
        logsnr_t,
        logsnr_s,
        model_output=None,
    ):
        """p(z_s | z_t)."""
        assert logsnr_t.shape == logsnr_s.shape == (z_t.shape[0],)
        model_output = self._run_model(net=net, z=z_t, logsnr=logsnr_t)

        logsnr_t = broadcast_from_left(logsnr_t, z_t.shape)
        logsnr_s = broadcast_from_left(logsnr_s, z_t.shape)

        pred_x = model_output['model_x']

        out = diffusion_reverse(
            z_t=z_t,
            logsnr_t=logsnr_t,
            logsnr_s=logsnr_s,
            x=pred_x,
            x_logvar='large',
        )
        out['pred_x'] = pred_x
        return out

    def vb(self, *, net, x, z_t, logsnr_t, logsnr_s, model_output):
        assert x.shape == z_t.shape
        assert logsnr_t.shape == logsnr_s.shape == (z_t.shape[0],)
        q_dist = diffusion_reverse(
            x=x,
            z_t=z_t,
            logsnr_t=broadcast_from_left(logsnr_t, x.shape),
            logsnr_s=broadcast_from_left(logsnr_s, x.shape),
            x_logvar='small',
        )
        p_dist = self.predict(
            net=net,
            z_t=z_t,
            logsnr_t=logsnr_t,
            logsnr_s=logsnr_s,
            model_output=model_output,
        )
        kl = normal_kl(
            mean1=q_dist['mean'],
            logvar1=q_dist['logvar'],
            mean2=p_dist['mean'],
            logvar2=p_dist['logvar'],
        )
        return mean_flat(kl) / np.log(2.0)

    def training_losses(self, *, net, x):
        assert x.dtype in [torch.float32, torch.float64]
        eps = torch.randn(x.shape, device=x.device, dtype=x.dtype)
        bc = lambda z: broadcast_from_left(z, x.shape)

        # sample logsnr
        if self.teacher_ddim is not None and self.teacher_mode == 'step2':
            # use discrete for distillation
            assert self.num_steps >= 1
            i = torch.randint(self.num_steps, (x.shape[0],), device=x.device)
            u = (i + 1).to(x.dtype) / self.num_steps
        else:
            # continuous time
            u = torch.rand(size=(x.shape[0],), dtype=x.dtype, device=x.device)
        logsnr = self.logsnr_schedule_fn(u)
        assert logsnr.shape == (x.shape[0],)

        # sample z ~ q(z_logsnr | x)
        z_dist = diffusion_forward(x=x, logsnr=bc(logsnr))
        z_t = z_dist['mean'] + z_dist['std'] * eps

        cond_w = None

        # get denoising target
        if self.teacher_ddim is not None:  # distillation mode
            cond_w = 4.0 * torch.rand_like(u)
            teacher_ddim = partial(self.teacher_ddim, guide=net.keywords['guide'])
            u_s = u - 1.0 / self.num_steps

            if self.teacher_mode == 'step1':
                # clone the teacher, which may be
                _, x_target, eps_target = teacher_ddim(
                    z_t=z_t, u_t=u, u_s=u_s, cond_w=cond_w
                )
            else:
                # two forward steps of DDIM from z_t using teacher
                u_mid = u - 0.5 / self.num_steps
                z_mid, _, __ = teacher_ddim(z_t=z_t, u_t=u, u_s=u_mid, cond_w=cond_w)
                z_teacher, x_pred_teacher, _ = teacher_ddim(
                    z_t=z_mid, u_t=u_mid, u_s=u_s, cond_w=cond_w
                )

                # get x-target implied by z_teacher (!= x_pred)
                logsnr_s = self.logsnr_schedule_fn(u_s)
                alpha_s = bc(torch.sqrt(torch.sigmoid(logsnr_s)))
                alpha_t = bc(torch.sqrt(torch.sigmoid(logsnr)))
                stdv_frac = bc(
                    torch.exp(0.5 * (F.softplus(logsnr) - F.softplus(logsnr_s)))
                )
                x_target = (z_teacher - stdv_frac * z_t) / (
                    alpha_s - stdv_frac * alpha_t
                )
                x_target = torch.where(bc(i == 0), x_pred_teacher, x_target)
                eps_target = predict_eps_from_x(z=z_t, x=x_target, logsnr=logsnr)

        else:  # denoise to original data
            x_target = x
            eps_target = eps

        # denoising loss
        # breakpoint() # feed in w to model if teacher_mode is on
        model_output = self._run_model(net=net, z=z_t, logsnr=logsnr)

        # so for the actual loss, no matter what the model predicts, we are going to
        # you could also get the target for v, but this is what they tend to use in their codebase

        x_mse = mean_flat(torch.square(model_output['model_x'] - x_target))
        eps_mse = mean_flat(torch.square(model_output['model_eps'] - eps_target))

        if self.loss_weight_type == 'snr_trunc':  # x_mse * max(SNR, 1)
            loss = torch.maximum(x_mse, eps_mse)
        elif self.loss_weight_type == 'snr':  # SNR * x_mse = eps_mse
            loss = eps_mse
        return {'loss': loss}

    def ddim_step(self, *, net, logsnr_t, logsnr_s, z_t, cond_w=None):
        bc = lambda z: broadcast_from_left(z, z_t.shape[:1])
        fbc = lambda z: broadcast_from_left(z, z_t.shape)
        model_out = self._run_model(
            net=net,
            z=z_t,
            logsnr=bc(logsnr_t),
        )
        x_pred_t = model_out['model_x']
        eps_pred_t = model_out['model_eps']

        if cond_w is not None:
            # run the model uncoditioned
            uncond_net = partial(net, guide=-torch.ones_like(net.keywords['guide']))
            uncond_model_eps = self._run_model(
                net=uncond_net, z=z_t, logsnr=bc(logsnr_t)
            )['model_eps']
            # we can do the combination in v-space or e-space, but we choose to do it in e-space.
            eps_pred_t = ((1 + cond_w) * eps_pred_t) - (cond_w * uncond_model_eps)
            x_pred_t = predict_x_from_eps(z=z_t, eps=eps_pred_t, logsnr=logsnr_t)
            # clip x and redo eps
            model_x = torch.clip(model_x, -1.0, 1.0)
            eps_pred_t = predict_eps_from_x(z=z_t, x=model_x, logsnr=logsnr_t)

        stdv_s = fbc(torch.sqrt(torch.sigmoid(-logsnr_s)))
        alpha_s = fbc(torch.sqrt(torch.sigmoid(logsnr_s)))
        z_s_pred = alpha_s * x_pred_t + stdv_s * eps_pred_t
        return z_s_pred, x_pred_t, eps_pred_t

    def reverse_dpm_step(self, net, logsnr_t, logsnr_s, z_t):
        shape, dtype = z_t.shape, z_t.dtype
        z_s_dist = self.predict(
            net=net,
            z_t=z_t,
            logsnr_t=torch.full((shape[0],), logsnr_t, dtype=dtype, device=z_t.device),
            logsnr_s=torch.full((shape[0],), logsnr_s, dtype=dtype, device=z_t.device),
        )
        eps = torch.randn(size=shape, dtype=dtype, device=z_t.device)
        z_s_pred = z_s_dist['mean'] + z_s_dist['std'] * eps
        x_pred_t = z_s_dist['pred_x']
        return x_pred_t, z_s_pred

    def sample(self, *, net, init_x):
        fbc = lambda z: broadcast_from_left(z, init_x.shape)
        if self.sampler == 'ddim':
            body_fun = lambda logsnr_t, logsnr_s, z_t: self.ddim_step(
                net=net,
                logsnr_t=logsnr_t,
                logsnr_s=logsnr_s,
                z_t=z_t,
                cond_w=self.cf_cond_w,
            )
        elif self.sampler == 'noisy':
            breakpoint()  # not supported rn
            body_fun = lambda logsnr_t, logsnr_s, z_t: self.reverse_dpm_step(
                net,
                logsnr_t=logsnr_t,
                logsnr_s=logsnr_s,
                z_t=z_t,
            )
        else:
            raise NotImplementedError(self.sampler)

        # loop over t = num_steps-1, ..., 0
        all_zs = []
        all_xs = []
        all_eps = []
        z_t = init_x
        for i in range(0, self.num_steps)[::-1]:
            torch_i = torch.tensor(i, device=init_x.device)
            logsnr_t = self.logsnr_schedule_fn((torch_i + 1.0) / self.num_steps)
            logsnr_s = self.logsnr_schedule_fn(torch_i / self.num_steps)
            z_s_pred, x_pred_t, eps_pred_t = body_fun(logsnr_t, logsnr_s, z_t)
            z_t = torch.where(fbc(torch_i) == 0, x_pred_t, z_s_pred)
            all_zs.append(z_t)
            all_xs.append(x_pred_t)
            all_eps.append(eps_pred_t)
        return torch.stack(all_zs), torch.stack(all_xs), torch.stack(all_eps)
