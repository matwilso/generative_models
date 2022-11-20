import functools
import math

import numpy as np
import torch
import torch.nn.functional as F


def get_timestep_embedding(
    timesteps, embedding_dim, max_time=1000.0, dtype=torch.float32
):
    """Get timestep embedding."""
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    timesteps *= 1000.0 / max_time

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = torch.concatenate([torch.sin(emb), torch.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def nearest_neighbor_upsample(x):
    B, H, W, C = x.shape  # pylint: disable=invalid-name
    x = x.reshape(B, H, 1, W, 1, C)
    x = torch.broadcast_to(x, (B, H, 2, W, 2, C))
    return x.reshape(B, H * 2, W * 2, C)


def diffusion_reverse(*, x, z_t, logsnr_s, logsnr_t, x_logvar):
    """q(z_s | z_t, x) (requires logsnr_s > logsnr_t (i.e. s < t))."""
    alpha_st = torch.sqrt((1.0 + torch.exp(-logsnr_t)) / (1.0 + torch.exp(-logsnr_s)))
    alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
    r = torch.exp(logsnr_t - logsnr_s)  # SNR(t)/SNR(s)
    one_minus_r = -torch.expm1(logsnr_t - logsnr_s)  # 1-SNR(t)/SNR(s)
    log_one_minus_r = log1mexp(logsnr_s - logsnr_t)  # log(1-SNR(t)/SNR(s))

    mean = r * alpha_st * z_t + one_minus_r * alpha_s * x

    if x_logvar == 'small':
        # same as setting x_logvar to -infinity
        var = one_minus_r * torch.sigmoid(-logsnr_s)
        logvar = log_one_minus_r + F.logsigmoid(-logsnr_s)
    elif x_logvar == 'large':
        # same as setting x_logvar to F.logsigmoid(-logsnr_t)
        var = one_minus_r * torch.sigmoid(-logsnr_t)
        logvar = log_one_minus_r + F.logsigmoid(-logsnr_t)
    elif x_logvar.startswith('medium:'):
        _, frac = x_logvar.split(':')
        frac = float(frac)
        assert 0 <= frac <= 1
        min_logvar = log_one_minus_r + F.logsigmoid(-logsnr_s)
        max_logvar = log_one_minus_r + F.logsigmoid(-logsnr_t)
        logvar = frac * max_logvar + (1 - frac) * min_logvar
        var = torch.exp(logvar)
    else:
        raise NotImplementedError(x_logvar)
    return {'mean': mean, 'std': torch.sqrt(var), 'var': var, 'logvar': logvar}


def diffusion_forward(*, x, logsnr):
    """q(z_t | x)."""
    assert x.shape == logsnr.shape
    return {
        'mean': x * torch.sqrt(torch.sigmoid(logsnr)),
        'std': torch.sqrt(torch.sigmoid(-logsnr)),
        'var': torch.sigmoid(-logsnr),
        'logvar': F.logsigmoid(-logsnr),
    }


def predict_x_from_eps(*, z, eps, logsnr):
    """x = (z - sigma*eps)/alpha."""
    logsnr = broadcast_from_left(logsnr, z.shape)
    assert z.shape == eps.shape == logsnr.shape
    return torch.sqrt(1.0 + torch.exp(-logsnr)) * (
        z - eps * torch.rsqrt(1.0 + torch.exp(logsnr))
    )


def predict_eps_from_x(*, z, x, logsnr):
    """eps = (z - alpha*x)/sigma."""
    logsnr = broadcast_from_left(logsnr, z.shape)
    assert z.shape == x.shape == logsnr.shape
    return torch.sqrt(1.0 + torch.exp(logsnr)) * (
        z - x * torch.rsqrt(1.0 + torch.exp(-logsnr))
    )


def predict_v_from_x_and_eps(*, x, eps, logsnr):
    logsnr = broadcast_from_left(logsnr, x.shape)
    alpha_t = torch.sqrt(torch.sigmoid(logsnr))
    sigma_t = torch.sqrt(torch.sigmoid(-logsnr))
    return alpha_t * eps - sigma_t * x


def predict_x_from_v(*, z, v, logsnr):
    logsnr = broadcast_from_left(logsnr, z.shape)
    alpha_t = torch.sqrt(torch.sigmoid(logsnr))
    sigma_t = torch.sqrt(torch.sigmoid(-logsnr))
    return alpha_t * z - sigma_t * v


def log1mexp(x, expm1_guard=1e-7):
    # taken from one of the links here: https://github.com/pytorch/pytorch/issues/39242
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    t = x < math.log(0.5)
    y = torch.zeros_like(x)
    y[t] = torch.log1p(-x[t].exp())

    # for x close to 0 we need expm1 for numerically stable computation
    # we furtmermore modify the backward pass to avoid instable gradients,
    # ie situations where the incoming output gradient is close to 0 and the gradient of expm1 is very large
    expxm1 = torch.expm1(x[~t])
    log1mexp_fw = (-expxm1).log()
    log1mexp_bw = (-expxm1 + expm1_guard).log()  # limits magnitude of gradient

    y[~t] = log1mexp_fw.detach() + (log1mexp_bw - log1mexp_bw.detach())
    return y


def broadcast_from_left(x, shape):
    if isinstance(x, float):
        x = torch.tensor(x, device='cuda')
    assert len(shape) >= x.ndim
    return torch.broadcast_to(x.reshape(x.shape + (1,) * (len(shape) - x.ndim)), shape)


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"
    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for  torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


# LogSNR schedules (t==0 => logsnr_max, t==1 => logsnr_min)


def _logsnr_schedule_uniform(t, *, logsnr_min, logsnr_max):
    return logsnr_min * t + logsnr_max * (1.0 - t)


def _onp_softplus(x):
    return np.logaddexp(x, 0)


def _logsnr_schedule_beta_const(t, *, logsnr_min, logsnr_max):
    b = _onp_softplus(-logsnr_max)
    a = _onp_softplus(-logsnr_min) - b
    return -torch.log(torch.expm1(a * t + b))


def _logsnr_schedule_beta_linear(t, *, logsnr_min, logsnr_max):
    b = _onp_softplus(-logsnr_max)
    a = _onp_softplus(-logsnr_min) - b
    return -torch.log(torch.expm1(a * t**2 + b))


def _logsnr_schedule_beta_interpolated(t, *, betas):
    betas = np.asarray(betas, dtype=np.float64)
    assert betas.ndim == 1
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    logsnr = np.log(alphas_cumprod) - np.log1p(-alphas_cumprod)
    return torch.interp(t, np.linspace(0, 1, len(betas)), logsnr)


def _logsnr_schedule_cosine(t, *, logsnr_min, logsnr_max):
    b = np.arctan(np.exp(-0.5 * logsnr_max))
    a = np.arctan(np.exp(-0.5 * logsnr_min)) - b
    return -2.0 * torch.log(torch.tan(a * t + b))


def _logsnr_schedule_iddpm_cosine_interpolated(t, *, num_timesteps):
    steps = np.arange(num_timesteps + 1, dtype=np.float64) / num_timesteps
    alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
    betas = np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
    return _logsnr_schedule_beta_interpolated(t, betas=betas)


def _logsnr_schedule_iddpm_cosine_respaced(t, *, num_timesteps, num_respaced_timesteps):
    """Improved DDPM respaced discrete time cosine schedule."""
    # original schedule
    steps = np.arange(num_timesteps + 1, dtype=np.float64) / num_timesteps
    alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
    betas = np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)

    # respace the schedule
    respaced_inds = np.round(
        np.linspace(0, 1, num_respaced_timesteps) * (num_timesteps - 1)
    ).astype(int)
    alpha_bar = np.cumprod(1.0 - betas)[respaced_inds]
    assert alpha_bar.shape == (num_respaced_timesteps,)
    logsnr = np.log(alpha_bar) - np.log1p(-alpha_bar)
    return torch.interp(t, np.linspace(0, 1, len(logsnr)), logsnr)


def get_logsnr_schedule(name, **kwargs):
    """Get log SNR schedule (t==0 => logsnr_max, t==1 => logsnr_min)."""
    schedules = {
        'uniform': _logsnr_schedule_uniform,
        'beta_const': _logsnr_schedule_beta_const,
        'beta_linear': _logsnr_schedule_beta_linear,
        'beta_interp': _logsnr_schedule_beta_interpolated,
        'cosine': _logsnr_schedule_cosine,
        'iddpm_cosine_interp': _logsnr_schedule_iddpm_cosine_interpolated,
        'iddpm_cosine_respaced': _logsnr_schedule_iddpm_cosine_respaced,
    }
    return functools.partial(schedules[name], **kwargs)
