import torch
import numpy as np
from tqdm import trange
from functools import partial

from cpd.samplers.diffusion import DiffusionSamplerWrapper
from cpd.samplers.ddim import DDIMSampler
from cpd.samplers.k_diffusion import KDiffusionSampler
from cpd.samplers.registry import register
from cpd.samplers.extension import create as create_extension
from cpd.util import safe_to

@register("Euler")
class EulerSamplerWrapper(DiffusionSamplerWrapper):
    def __init__(self, name, **kwargs):
        kwargs["constructor"] = EulerDiffusionSampler
        super().__init__(name, **kwargs)
class EulerDiffusionSampler(KDiffusionSampler):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    def __init__(self, model):
        super().__init__(model, "euler")

    @torch.no_grad()
    def _sampling(self, x, sigmas, model_args=None, **kwargs):
        callback = kwargs.get("callback", None) 
        disable = kwargs.get("disable", None) 
        clip_sample = kwargs.get("clip_sample", False)
        clip_sample_alg = kwargs.get("clip_sample_alg", "dynamic_thresholding")
        clip_sample_thresh = kwargs.get("clip_sample_thresh", 90)
        if clip_sample:
            sample_thresholding = create_extension(clip_sample_alg) 

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
            x = safe_to(x, device=eps.device)
            d = to_ode(x, sigma_hat, eps)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'eps': eps})
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
            if clip_sample:
                x = sample_thresholding.apply(x, 0, threshold=clip_sample_thresh)
        return x
        

@register("Euler Ancestral")
class EulerAncestralSamplerWrapper(DiffusionSamplerWrapper):
    def __init__(self, name, **kwargs):
        kwargs["constructor"] = EulerAncestralDiffusionSampler
        super().__init__(name, **kwargs)
class EulerAncestralDiffusionSampler(KDiffusionSampler):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    def __init__(self, model):
        super().__init__(model, "euler ancestral")

    @torch.no_grad()
    def _sampling(self, x, sigmas, model_args=None, **kwargs):
        callback = kwargs.get("callback", None) 
        disable = kwargs.get("disable", None)
        clip_sample = kwargs.get("clip_sample", False)
        clip_sample_alg = kwargs.get("clip_sample_alg", "dynamic_thresholding")
        clip_sample_thresh = kwargs.get("clip_sample_thresh", 90)
        if clip_sample:
            sample_thresholding = create_extension(clip_sample_alg) 
        model_args = {} if model_args is None else model_args
        s_in = x.new_ones([x.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            model_args["t_idx"] = i
            eps = self.denoiser(x, sigmas[i] * s_in, **model_args)
            x = safe_to(x, device=eps.device)
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'eps': eps})
            d = to_ode(x, sigmas[i], eps)
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt
            x = x + torch.randn_like(x) * sigma_up
            if clip_sample:
                x = sample_thresholding.apply(x, 0, threshold=clip_sample_thresh)
        return x
        
def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up
def to_ode(x, sigma, eps):
    """Convert a denoiser output to a Karras ODE derivative."""
    return (x - eps) / append_dims(sigma, x.ndim)

def append_dims(x, target_dims):
    """Append dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]