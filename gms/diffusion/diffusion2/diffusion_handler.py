import numpy as np
import torch

from gms.diffusion.diffusion2.diffusion_utils import (
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


class DiffusionHandler:
    def __init__(
        self, *, mean_type, num_steps, distillation_target=None, sampler='ddim'
    ):
        self.mean_type = mean_type
        self.num_steps = num_steps
        self.distillation_target = distillation_target
        self.logsnr_schedule_fn = get_logsnr_schedule(
            'cosine', logsnr_min=-20.0, logsnr_max=20.0
        )
        self.sampler = sampler

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
        if self.distillation_target is not None:
            # use discrete for distillation
            assert self.num_steps >= 1
            i = torch.randint(
                size=(x.shape[0],), minval=0, maxval=self.num_steps, device=x.device
            )
            u = (i + 1).astype(x.dtype) / self.num_steps
        else:
            # continuous time
            u = torch.rand(size=(x.shape[0],), dtype=x.dtype, device=x.device)
        logsnr = self.logsnr_schedule_fn(u)
        assert logsnr.shape == (x.shape[0],)

        # sample z ~ q(z_logsnr | x)
        z_dist = diffusion_forward(x=x, logsnr=bc(logsnr))
        z = z_dist['mean'] + z_dist['std'] * eps

        # get denoising target
        if self.distillation_target is not None:  # distillation
            # two forward steps of DDIM from z_t using teacher
            teach_out_start = self._run_model(
                net=self.distillation_target, z=z, logsnr=logsnr
            )
            x_pred = teach_out_start['model_x']
            eps_pred = teach_out_start['model_eps']

            u_mid = u - 0.5 / self.num_steps
            logsnr_mid = self.logsnr_schedule_fn(u_mid)
            stdv_mid = bc(torch.sqrt(torch.sigmoid(-logsnr_mid)))
            a_mid = bc(torch.sqrt(torch.sigmoid(logsnr_mid)))
            z_mid = a_mid * x_pred + stdv_mid * eps_pred

            teach_out_mid = self._run_model(
                net=self.distillation_target,
                z=z_mid,
                logsnr=logsnr_mid,
            )
            x_pred = teach_out_mid['model_x']
            eps_pred = teach_out_mid['model_eps']

            u_s = u - 1.0 / self.num_steps
            logsnr_s = self.logsnr_schedule_fn(u_s)
            stdv_s = bc(torch.sqrt(torch.sigmoid(-logsnr_s)))
            a_s = bc(torch.sqrt(torch.sigmoid(logsnr_s)))
            z_teacher = a_s * x_pred + stdv_s * eps_pred

            # get x-target implied by z_teacher (!= x_pred)
            a_t = bc(torch.sqrt(torch.sigmoid(logsnr)))
            stdv_frac = bc(
                torch.exp(0.5 * (torch.softplus(logsnr) - torch.softplus(logsnr_s)))
            )
            x_target = (z_teacher - stdv_frac * z) / (a_s - stdv_frac * a_t)
            x_target = torch.where(bc(i == 0), x_pred, x_target)
            eps_target = predict_eps_from_x(z=z, x=x_target, logsnr=logsnr)

        else:  # denoise to original data
            x_target = x
            eps_target = eps

        # denoising loss
        model_output = self._run_model(net=net, z=z, logsnr=logsnr)

        # so for the actual loss, no matter what the model predicts, we are going to
        # you could also get the target for v, but this is what they tend to use in their codebase

        x_mse = mean_flat(torch.square(model_output['model_x'] - x_target))
        eps_mse = mean_flat(torch.square(model_output['model_eps'] - eps_target))

        # x_mse * max(SNR, 1). SNR-trunc
        loss = torch.maximum(x_mse, eps_mse)
        return {'loss': loss}

    def ddim_step(self, net, i, z_t):
        shape, dtype = z_t.shape, z_t.dtype
        logsnr_t = self.logsnr_schedule_fn((i + 1.0) / self.num_steps)
        logsnr_s = self.logsnr_schedule_fn(i / self.num_steps)
        model_out = self._run_model(
            net=net,
            z=z_t,
            logsnr=torch.full((shape[0],), logsnr_t, dtype=dtype, device=z_t.device),
        )
        x_pred_t = model_out['model_x']
        eps_pred_t = model_out['model_eps']
        stdv_s = torch.sqrt(torch.sigmoid(-logsnr_s))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        z_s_pred = alpha_s * x_pred_t + stdv_s * eps_pred_t
        return torch.where(i == 0, x_pred_t, z_s_pred)

    def reverse_dpm_step(self, net, i, z_t):
        shape, dtype = z_t.shape, z_t.dtype
        logsnr_t = self.logsnr_schedule_fn((i + 1.0) / self.num_steps)
        logsnr_s = self.logsnr_schedule_fn(i / self.num_steps)
        z_s_dist = self.predict(
            net=net,
            z_t=z_t,
            logsnr_t=torch.full((shape[0],), logsnr_t, dtype=dtype, device=z_t.device),
            logsnr_s=torch.full((shape[0],), logsnr_s, dtype=dtype, device=z_t.device),
        )
        eps = torch.randn(size=shape, dtype=dtype, device=z_t.device)
        return torch.where(
            i == 0, z_s_dist['pred_x'], z_s_dist['mean'] + z_s_dist['std'] * eps
        )

    def sample(self, *, net, init_x):
        if self.sampler == 'ddim':
            body_fun = lambda i, z_t: self.ddim_step(
                net,
                i,
                z_t,
            )
        elif self.sampler == 'noisy':
            body_fun = lambda i, z_t: self.reverse_dpm_step(
                net,
                i,
                z_t,
            )
        else:
            raise NotImplementedError(self.sampler)

        # loop over t = num_steps-1, ..., 0
        outs = []
        z_t = init_x
        for i in range(0, self.num_steps)[::-1]:
            torch_i = torch.tensor(i, device=init_x.device)
            z_t = body_fun(torch_i, z_t)
            outs.append(z_t)
        return torch.stack(outs)
