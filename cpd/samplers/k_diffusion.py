import math
import traceback
from functools import partial

import torch

import numpy as np
from scipy import integrate
from tqdm import trange

from cpd.util import CudaMon, safe_to
from cpd.samplers.diffusion import DiffusionSamplerWrapper
from cpd.samplers.registry import register
from cpd.samplers.extension.callbacks import render_callback
from cpd.samplers.extension import create as create_extension
from cpd.scheduler.ddim import DiscreteScheduler
from cpd.samplers.extension.denoiser import Denoiser

from IPython import display
from PIL import Image

class KDiffusionSampler:
    def __init__(self, model, name='sample_heun'):
        self.name = name
        self.denoiser = Denoiser(model["unet"], 
                                 model["vae"], 
                                 model["tokenizer"], 
                                 model["clip_new_model"], 
                                 model["decode"])
    
    def sample_img2img(self, x, noise, steps, **kwargs):
        denoising_strength = kwargs.get("denoising_strength", 0.)
        verbose = kwargs.get("verbose", False)
        disabled = kwargs.get("silent", False)
        scheduler = kwargs.get("scheduler", "default")
        
        t_enc = int((1 - min(denoising_strength, 0.999)) * steps)
        
        sigmas = self.denoiser.scheduler.get_sigmas(scheduler, steps, **kwargs)

        noise = noise * sigmas[steps - t_enc - 1]

        xi = x + noise

        sigma_sched = sigmas[steps - t_enc - 1:]

        kwargs["total_steps"] = len(sigma_sched)

        samples = self._sampling(xi, sigma_sched, 
                            model_args=kwargs, 
                            disable=not verbose,
                            **kwargs)
        #return self.func(self.model_wrap_cfg, xi, sigma_sched, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': p.cfg_scale}, disable=False, callback=self.callback_state)
        return samples

    def sample(self, steps, batch_size, shape, **kwargs):        
        verbose = kwargs.get("verbose", False)
        disabled = kwargs.get("silent", False)
        decode = kwargs.get("decode", False)
        
        x_T = kwargs.get("x_T", None)
        scheduler = kwargs.get("scheduler", "default")
        sigmas = self.denoiser.scheduler.get_sigmas(scheduler, steps, **kwargs)
        if decode:
            denoising_strength = kwargs.get("denoising_strength", 0.)
            t_enc = int((1 - min(denoising_strength, 0.999)) * steps)
            sigmas = sigmas[steps - t_enc - 1:]
            noise = torch.randn([batch_size]+shape).cuda()
            noise = noise * sigmas[0]
            x = x_T + noise
        else:
            if x_T is None:
                x_T = torch.randn([batch_size]+shape).cuda()
            x = x_T * sigmas[0]

        kwargs["total_steps"] = len(sigmas)

        samples = self._sampling(x, sigmas, 
                            model_args=kwargs, 
                            disable=disabled,
                            **kwargs)
        return samples

    def _sampling(self, x, sigmas, model_args=None, **kwargs):
        raise NotImplementedError()

    @torch.no_grad()
    def stochastic_encode(self, x0, t, noise=None, **kwargs):
        scheduler = kwargs.get("scheduler", "default")
        sigmas = self.denoiser.scheduler.get_sigmas(scheduler, steps, **kwargs)
        if noise is None:
            noise = torch.randn_like(x0)
        a_t = safe_to(self.denoiser.scheduler.alphas_cumprod[t]).sqrt()
        sqrt_one_minus_at = safe_to(self.denoiser.scheduler.sqrt_one_minus_alphas_cumprod[t])
        self.clog("stochastic_encode", "done")
        return (a_t * x0 +
                sqrt_one_minus_at * noise)
