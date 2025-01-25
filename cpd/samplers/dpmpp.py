import torch
import numpy as np
from tqdm import trange
from functools import partial

from cpd.samplers.diffusion import DiffusionSamplerWrapper
from cpd.samplers.k_diffusion import KDiffusionSampler
from cpd.samplers.registry import register
from cpd.samplers.extension import create as create_extension
from cpd.util import safe_to

@register("DPM++ 2m")
class DPMPlusPlus2mSamplerWrapper(DiffusionSamplerWrapper):
    def __init__(self, name, **kwargs):
        kwargs["constructor"] = DPMPlusPlus2mDiffusionSampler
        super().__init__(name, **kwargs)
class DPMPlusPlus2mDiffusionSampler(KDiffusionSampler):
    """A sampler inspired by DPM-Solver++(2M)."""
    
    def __init__(self, model):
        super().__init__(model, "dpmpp 2m")
    @torch.no_grad()
    def _sampling(self, x, sigmas, model_args=None, **kwargs):
        callback = kwargs.get("callback", None) 
        disable = kwargs.get("disable", None)         
        model_args = {} if model_args is None else model_args
        s_in = x.new_ones([x.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        old_eps = None
        clip_sample = kwargs.get("clip_sample", False)
        clip_sample_alg = kwargs.get("clip_sample_alg", "dynamic_thresholding")
        clip_sample_thresh = kwargs.get("clip_sample_thresh", 90)
        if clip_sample:
            sample_thresholding = create_extension(clip_sample_alg) 

        for i in trange(len(sigmas) - 1, disable=disable):
            model_args["t_idx"] = i
            eps = self.denoiser(x, sigmas[i] * s_in, **model_args)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'eps': eps})
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            if old_eps is None or sigmas[i + 1] == 0:
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * eps
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                eps_d = (1 + 1 / (2 * r)) * eps - (1 / (2 * r)) * old_eps
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * eps_d
            if clip_sample:
                x = sample_thresholding.apply(x, 0, threshold=clip_sample_thresh)
                                   
            old_eps = eps
            
        return x        

@register("DPM++ 2s Ancestral")
class DPMPlusPlus2sAncestralSamplerWrapper(DiffusionSamplerWrapper):
    def __init__(self, name, **kwargs):
        kwargs["constructor"] = DPMPlusPlus2sAncestralDiffusionSampler
        super().__init__(name, **kwargs)
class DPMPlusPlus2sAncestralDiffusionSampler(KDiffusionSampler):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    
    def __init__(self, model):
        super().__init__(model, "dpmpp 2s ancestral")
    
    @torch.no_grad()
    def _sampling(self, x, sigmas, model_args=None, **kwargs):
        callback = kwargs.get("callback", None) 
        disable = kwargs.get("disable", None) 
        
        eta = kwargs.get("eta", 1.)
        tmp = kwargs.get("temperature", 1.)
       
        model_args = {} if model_args is None else model_args

        s_in = x.new_ones([x.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        clip_sample = kwargs.get("clip_sample", False)
        clip_sample_alg = kwargs.get("clip_sample_alg", "dynamic_thresholding")
        clip_sample_thresh = kwargs.get("clip_sample_thresh", 90)
        if clip_sample:
            sample_thresholding = create_extension(clip_sample_alg) 

        for i in trange(len(sigmas) - 1, disable=disable):
            eps = self.denoiser(x, sigmas[i] * s_in, **model_args)
            if clip_sample:
                x = sample_thresholding.apply(x, 0, threshold=clip_sample_thresh)  
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'eps': eps})
            if sigma_down == 0:
                # Euler method
                d = to_ode(x, sigmas[i], eps)
                dt = sigma_down - sigmas[i]
                x = x + d * dt                             
            else:
                # DPM-Solver-2++(2S)
                t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * eps
                eps_2 = self.denoiser(x_2, sigma_fn(s) * s_in, **model_args)
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * eps_2
                
            # Noise addition
            x = x + torch.randn_like(x) * tmp * sigma_up
        return x       

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def append_dims(x, target_dims):
    """Append dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def to_ode(x, sigma, eps):
    """Convert a denoiser output to a Karras ODE derivative."""
    return (x - eps) / append_dims(sigma, x.ndim)