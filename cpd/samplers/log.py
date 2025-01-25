import torch
from torchdiffeq import odeint
import numpy as np
from tqdm import trange
from functools import partial

from cpd.samplers.diffusion import DiffusionSamplerWrapper
from cpd.samplers.ddim import DDIMSampler
from cpd.samplers.k_diffusion import KDiffusionSampler
from cpd.samplers.registry import register

@register("Log")
class LogLikelihoodSamplerWrapper(DiffusionSamplerWrapper):
    def __init__(self, name, **kwargs):
        kwargs["constructor"] = LogLikelihoodDiffusionSampler
        super().__init__(name, **kwargs)

class LogLikelihoodDiffusionSampler(KDiffusionSampler):
    """Implements single step log likelihood ODE solver."""
    
    def __init__(self, model):
        super().__init__(model, "log")
    
    @torch.no_grad()
    def _sampling(self, x, sigmas, model_args=None, **kwargs):
        callback = kwargs.get("callback", None) 
        disable = kwargs.get("disable", None) 
        s_min=kwargs.get("s_tmin", 0.) 
        s_max=kwargs.get("s_tmax", float('inf'))
        atol = kwargs.get("atol", 1e-4)
        rtol = kwargs.get("rtol", 1e-4)
        model_args = {} if model_args is None else model_args
        s_in = x.new_ones([x.shape[0]])
        
        v = torch.randint_like(x, 2) * 2 - 1
        fevals = 0
        def ode_fn(sigma, x):
            nonlocal fevals
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()
                eps = self.denoiser(x, sigma * s_in, **model_args)
                d = to_ode(x, sigma, eps)
                fevals += 1
                grad = torch.autograd.grad((d * v).sum(), x)[0]
                d_ll = (v * grad).flatten(1).sum(1)
            return d.detach(), d_ll
        x_min = x, x.new_zeros([x.shape[0]])
        t = x.new_tensor([s_min, s_max])
        sol = odeint(ode_fn, x_min, t, atol=atol, rtol=rtol, method='dopri5')
        latent, delta_ll = sol[0][-1], sol[1][-1]
        ll_prior = torch.distributions.Normal(0, s_max).log_prob(latent).flatten(1).sum(1)
        x = ll_prior + delta_ll
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