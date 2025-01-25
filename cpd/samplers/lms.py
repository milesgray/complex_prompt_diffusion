
import torch
import torch.nn as nn
from scipy import integrate
from tqdm import trange

from cpd.util import CudaMon
from cpd.samplers.diffusion import DiffusionSamplerWrapper
from cpd.samplers.ddim import DDIMSampler
from cpd.samplers.k_diffusion import KDiffusionSampler
from cpd.samplers.registry import register

@register("LMS")
class LMSSamplerWrapper(DiffusionSamplerWrapper):
    def __init__(self, name, **kwargs):
        kwargs["constructor"] = LMSDiffusionSampler
        super().__init__(name, **kwargs)
class LMSDiffusionSampler(KDiffusionSampler):    
    """_summary_

    """
    
    def __init__(self, model):
        super().__init__(model, "lms")
    
    @torch.no_grad()
    def _sampling(self, x, sigmas, model_args=None, **kwargs):
        callback = kwargs.get("callback", None) 
        disable = kwargs.get("disable", None) 
        order = kwargs.get("order", 4)
        clip_sample = kwargs.get("clip_sample", False)
        clip_sample_thresh = kwargs.get("clip_sample_thresh", 90)

        model_args = {} if model_args is None else model_args
        s_in = x.new_ones([x.shape[0]])
        ds = []
        model_args["bar"] = trange(len(sigmas) - 1, disable=disable)
        for i in model_args["bar"]:
            model_args["t_idx"] = i
            eps = self.denoiser(x, sigmas[i] * s_in, **model_args)
            d = to_ode(x, sigmas[i], eps)
            ds.append(d)
            if len(ds) > order:
                ds.pop(0)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'eps': eps})
            cur_order = min(i + 1, order)
            coeffs = [linear_multistep_coeff(cur_order, sigmas.cpu(), i, j) for j in range(cur_order)]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
        return x

def linear_multistep_coeff(order, t, i, j):
    if order - 1 > i:
        raise ValueError(f'Order {order} too high for step {i}')
    def fn(tau):
        prod = 1.
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod
    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]

def to_ode(x, sigma, eps):
    """Convert a denoiser output to a Karras ODE derivative."""
    return (x - eps) / append_dims(sigma, x.ndim)

def append_dims(x, target_dims):
    """Append dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

   