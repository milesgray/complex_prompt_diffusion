import torch
import numpy as np
from tqdm import trange
from functools import partial

from cpd.samplers.diffusion import DiffusionSamplerWrapper
from cpd.samplers.k_diffusion import KDiffusionSampler
from cpd.samplers.registry import register

@register("DPM2")
class DPM2SamplerWrapper(DiffusionSamplerWrapper):
    def __init__(self, name, **kwargs):
        kwargs["constructor"] = DPM2DiffusionSampler
        super().__init__(name, **kwargs)
class DPM2DiffusionSampler(KDiffusionSampler):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    
    def __init__(self, model):
        super().__init__(model, "dpm2")
    
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
            # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
            sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            
            eps_2 = self.denoiser(x_2, sigma_mid * s_in, **model_args)
            d_2 = to_ode(x_2, sigma_mid, eps_2)
            x = x + d_2 * dt_2            
        return x

@register("DPM2 Ancestral")
class DPM2AncestralSamplerWrapper(DiffusionSamplerWrapper):
    def __init__(self, name, **kwargs):
        kwargs["constructor"] = DPM2AncestralDiffusionSampler
        super().__init__(name, **kwargs)
class DPM2AncestralDiffusionSampler(KDiffusionSampler):
    """Ancestral sampling with DPM-Solver inspired second-order steps."""
    
    def __init__(self, model):
        super().__init__(model, "dpm2 ancestral")
    
    @torch.no_grad()
    def _sampling(self, x, sigmas, model_args=None, **kwargs):
        callback = kwargs.get("callback", None) 
        disable = kwargs.get("disable", None) 
        clip_sample = kwargs.get("clip_sample", False)
        clip_sample_thresh = kwargs.get("clip_sample_thresh", 90)
       
        model_args = {} if model_args is None else model_args
        s_in = x.new_ones([x.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            eps = self.denoiser(x, sigmas[i] * s_in, **model_args)
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'eps': eps})
            d = to_ode(x, sigmas[i], eps)
            # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
            sigma_mid = ((sigmas[i] ** (1 / 3) + sigma_down ** (1 / 3)) / 2) ** 3
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigma_down - sigmas[i]
            x_2 = x + d * dt_1
            
            eps_2 = self.denoiser(x_2, sigma_mid * s_in, **model_args)
            d_2 = to_ode(x_2, sigma_mid, eps_2)
            x = x + d_2 * dt_2
            
            x = x + torch.randn_like(x) * sigma_up
        return x


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def append_dims(x, target_dims):
    """Append dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def to_ode(x, sigma, eps):
    """Convert a denoiser output to a Karras ODE derivative."""
    return (x - eps) / append_dims(sigma, x.ndim)        