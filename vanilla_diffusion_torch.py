# Training Loop & forward backward

# model

# diffusion

# diffusion compute losses

import copy
import os

import numpy as np
import torch
import torch.distributed as dist

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    noise = torch.randn_like(x_start)
    return (
        _extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + _extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        * noise
    )

def q_posterior_mean_variance(
    x_start, x_t, t, posterior_mean_coef1, posterior_mean_coef2, 
    posterior_variance, posterior_log_variance_clipped
    ):
    posterior_mean = (
        _extract_into_tensor(posterior_mean_coef1, t, x_t.shape) * x_start
        + _extract_into_tensor(posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance = _extract_into_tensor(posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped = _extract_into_tensor(
        posterior_log_variance_clipped, t, x_t.shape
    )
    return posterior_mean, posterior_variance, posterior_log_variance_clipped

def p_mean_variance(model, x, t, denoised_fn):

    B, C = x.shape[:2]
    model_output = model(x, t)
    model_output, model_var_values = torch.split(model_output, C, dim=1)
    model_log_variance = model_var_values
    model_variance = torch.exp(model_log_variance)
    pred_xstart = denoised_fn(model_output)
    model_mean, _, _ = q_posterior_mean_variance(pred_xstart, x, t)
    return {
        "mean": model_mean,
        "variance": model_variance,
        "log_variance": model_log_variance,
        "pred_xstart": pred_xstart,
        }

def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    return log_probs

def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
    
    true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance()
    out = self.p_mean_variance()
    kl = normal_kl()
    kl = mean_flat(kl) / np.log(2.0)

    decoder_nll = -discretized_gaussian_log_likelihood()
    decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

    output = torch.where((t == 0), decoder_nll, kl)
    return {"output": output, "pred_xstart": out["pred_xstart"]}

#def training_losses(self, model, x_start, t):

x_start = 

noise = torch.randn_like(x_start)
x_t = self.q_sample(x_start, t, noise=noise)

loss = self._vb_terms_bpd(model=model,x_start=x_start,x_t=x_t,t=t,clip_denoised=False)["output"]
model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

if self.model_var_type in [
    ModelVarType.LEARNED,
    ModelVarType.LEARNED_RANGE,
]:
    B, C = x_t.shape[:2]
    assert model_output.shape == (B, C * 2, *x_t.shape[2:])
    model_output, model_var_values = th.split(model_output, C, dim=1)
    # Learn the variance using the variational bound, but don't let
    # it affect our mean prediction.
    frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
    terms["vb"] = self._vb_terms_bpd(
        model=lambda *args, r=frozen_out: r,
        x_start=x_start,
        x_t=x_t,
        t=t,
        clip_denoised=False,
    )["output"]
    if self.loss_type == LossType.RESCALED_MSE:
        # Divide by 1000 for equivalence with initial implementation.
        # Without a factor of 1/1000, the VB term hurts the MSE term.
        terms["vb"] *= self.num_timesteps / 1000.0

target = {
    ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
        x_start=x_start, x_t=x_t, t=t
    )[0],
    ModelMeanType.START_X: x_start,
    ModelMeanType.EPSILON: noise,
}[self.model_mean_type]
assert model_output.shape == target.shape == x_start.shape
terms["mse"] = mean_flat((target - model_output) ** 2)
if "vb" in terms:
    terms["loss"] = terms["mse"] + terms["vb"]
else:
    terms["loss"] = terms["mse"]
else:
    raise NotImplementedError(self.loss_type)

return terms