import torch
import numpy as np
from tqdm import trange
from functools import partial

from cpd.samplers.diffusion import DiffusionSamplerWrapper
from cpd.samplers.ddim import DDIMSampler
from cpd.samplers.k_diffusion import KDiffusionSampler
from cpd.samplers.registry import register

@register("Huen")
class HeunSamplerWrapper(DiffusionSamplerWrapper):
    def __init__(self, name, **kwargs):
        kwargs["constructor"] = HeunDiffusionSampler
        super().__init__(name, **kwargs)

class HeunDiffusionSampler(KDiffusionSampler):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    
    def __init__(self, model):
        super().__init__(model, "euler ancestral")
    
    @torch.no_grad()
    def _sampling(self, x, sigmas, model_args=None, **kwargs):
        callback = kwargs.get("callback", None) 
        disable = kwargs.get("disable", None) 
        clip_sample = kwargs.get("clip_sample", False)
        clip_sample_thresh = kwargs.get("clip_sample_thresh", 90)
        s_churn= kwargs.get("s_churn", 0.) 
        s_tmin=kwargs.get("s_tmin", 0.) 
        s_tmax=kwargs.get("s_tmax", float('inf'))
        s_noise=kwargs.get("s_noise", 1.)
        model_args = {} if model_args is None else model_args
        s_in = x.new_ones([x.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            model_args["t_idx"] = i
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            eps = torch.randn_like(x) * s_noise
            sigma_hat = sigmas[i] * (gamma + 1)
            if gamma > 0:
                x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
            eps = self.denoiser(x, sigma_hat * s_in, **model_args)
            d = to_ode(x, sigma_hat, eps)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'eps': eps})
            dt = sigmas[i + 1] - sigma_hat
            if sigmas[i + 1] == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                eps_2 = self.denoiser(x_2, sigmas[i + 1] * s_in, **model_args)
                d_2 = to_ode(x_2, sigmas[i + 1], eps_2)
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
        return x

def to_ode(x, sigma, eps):
    """Convert a denoiser output to a Karras ODE derivative."""
    return (x - eps) / append_dims(sigma, x.ndim)       


def append_dims(x, target_dims):
    """Append dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append] 