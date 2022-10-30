"""
code forked from here: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
"""
import numpy as np
import torch
import torch.functional as F

from gms.diffusion.diffusion1.utils import (
    Extractable,
    discretized_gaussian_log_likelihood,
    get_cosine_beta_schedule,
    mean_flat,
    normal_kl,
)

# the usage of logsnr is quite nice actually. makes the equations simpler


def diffusion_forward(*, x, logsnr):
    """q(z_t | x)."""
    return {
        'mean': x * torch.sqrt(F.sigmoid(logsnr)),
        'std': torch.sqrt(F.sigmoid(-logsnr)),
        'var': F.sigmoid(-logsnr),
        'logvar': F.logsigmoid(-logsnr),
    }


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    This only implements RESCALED_MSE loss and 'linear' beta scheduling.
    """

    def __init__(self, num_timesteps):
        self.num_timesteps = (
            num_timesteps  # steps of diffusion (e.g., 1000 in Ho paper)
        )
        # initialize betas and calculate alphas for each timestep
        # beta = variance
        self.betas = get_cosine_beta_schedule(num_timesteps)
        self.log_betas = np.log(self.betas)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                self.__dict__[key] = Extractable(
                    val
                )  # make them easy to index by time and broadcast

    # FORWARD DIFFUSION
    def q_sample(self, x_start, t, noise):
        """Sample noise ~ q(x_t | x_0)"""
        return (self.sqrt_alphas_cumprod[t] * x_start) + (
            self.sqrt_one_minus_alphas_cumprod[t] * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Compute the diffusion posterior distribution: q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
            self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        )
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # REVERSE DIFFUSION
    def p_dist(self, model, x, t, model_kwargs={}):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        # Pass x_t through model and compute eps
        model_out = model(x, t, **model_kwargs)
        eps, model_var = torch.split(model_out, x.shape[1], dim=1)
        # Compute variance by interpolating between min and max values (equation 15, improved paper)
        min_log = self.posterior_log_variance_clipped[t]
        max_log = self.log_betas[t]
        frac = (model_var + 1) / 2  # The model_var is [-1, 1] for [min_var, max_var].
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)
        # Predict x_0 from eps. at any point, this is our best guess of the true output. it will vary and become refined as we reduce noise.
        pred_xstart = (
            self.sqrt_recip_alphas_cumprod[t] * x
            - self.sqrt_recipm1_alphas_cumprod[t] * eps
        )
        pred_xstart = pred_xstart.clamp(-1, 1)

        # using the reparamerization, the mean is defined via the posterior distribution equation. (see equation 13, improved paper)
        model_mean, _, _ = self.q_posterior(x_start=pred_xstart, x_t=x, t=t)
        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _p_sample_step(self, model, x, t, model_kwargs={}):
        """Sample x_{t-1} ~ p(x_{t-1}|x_t) using the model."""
        out = self.p_dist(model, x, t, model_kwargs=model_kwargs)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = (
            out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        )
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample(self, model, shape, noise=None, model_kwargs={}):
        """Generate a full sample from the model and keep history"""
        device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = noise if noise is not None else torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        outs = []
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self._p_sample_step(model, img, t, model_kwargs=model_kwargs)
                outs += [out]
                img = out["sample"]
        return outs

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            self.sqrt_recip_alphas_cumprod[t] * x_t - pred_xstart
        ) / self.sqrt_recipm1_alphas_cumprod[t]

    def _ddim_sample_step(
        self,
        model,
        x,
        t,
        model_kwargs={},
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as _p_sample_step().
        """
        out = self.p_dist(
            model,
            x,
            t,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod_prev[t]
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample(
        self,
        model,
        shape,
        noise=None,
        model_kwargs={},
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample().
        """
        device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = noise if noise is not None else torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        outs = []

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self._ddim_sample_step(
                    model,
                    img,
                    t,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                outs += [out]
                img = out["sample"]

        return outs

    # LOSS TERMS AND LOSSES

    def _vb_terms_bpd(self, model, x_start, x_t, t, model_kwargs={}):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        """
        # KL between posterior and learned model to inject info about data (L_{t-1} terms)
        true_mean, _, true_log_variance_clipped = self.q_posterior(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_dist(model, x_t, t, model_kwargs={})
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)
        # reconstruction (L_0 term)
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
        output = torch.where(
            (t == 0), decoder_nll, kl
        )  # use reconstruction loss only if t == 0
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs={}):
        """Compute training losses for an image and timesteps"""
        terms = {}
        # add diffusion noise according to the t values
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        # pass the noisy image to the model and learn to predict the noise and variance.
        model_output = model(x_t, t, **model_kwargs)
        eps, model_var = torch.split(model_output, x_t.shape[1], dim=1)
        terms["mse"] = mean_flat((noise - eps) ** 2)
        # learn the variance using the variational bound, but don't let it affect our mean prediction.
        frozen_out = torch.cat([eps.detach(), model_var], dim=1)
        terms["vb"] = self._vb_terms_bpd(
            model=lambda *args, r=frozen_out: r, x_start=x_start, x_t=x_t, t=t
        )["output"]
        terms["vb"] *= (
            self.num_timesteps / 1000.0
        )  # Divide by 1000 for equivalence with initial implementation. else VB term hurts the MSE term.
        terms["loss"] = terms["mse"] + terms["vb"]
        return terms
