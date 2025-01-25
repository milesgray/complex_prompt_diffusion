import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from cpd.util import safe_to
from cpd.samplers.diffusion import DiffusionSamplerWrapper
from cpd.samplers.ddim import DDIMSampler
from cpd.samplers.registry import register

@register("PLMS")
class PLMSSamplerWrapper(DiffusionSamplerWrapper):
    def __init__(self, name, **kwargs):
        kwargs["constructor"] = PLMSSampler
        super().__init__(name, **kwargs)


class PLMSSampler(DDIMSampler):
    @torch.no_grad()
    def p_sample(self, x, c, t, index, **kwargs):        
        old_eps = kwargs.get("old_eps", [])
        t_next = kwargs.get("t_next", t-1)

        e_t = self._calculate_epsilon(x, c, t, **kwargs)        
        e_t, x = self._get_clip_guide(x, t, c, e_t,  **kwargs)

        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = self._get_x_prev(x, e_t, index)
            e_t_next = self._calculate_epsilon(x_prev, c, t_next,  **kwargs)
            e_t_next, x = self._get_clip_guide(x, t, c, e_t_next, **kwargs)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24
      
        x_prev, pred_x0 = self._get_x_prev(x, e_t_prime, index)
        self.clog("p_sample", f"{t} done")
        return x_prev, pred_x0, e_t

    def _get_x_prev(self, x, e_t, index):
        # See formula (9) of PNDM paper https://arxiv.org/pdf/2202.09778.pdf
        # this function computes x_(t−δ) using the formula of (9)
        # Note that x_t needs to be added to both sides of the equation

        # Notation (<variable name> -> <name in paper>
        # alpha_prod_t -> α_t
        # alpha_prod_t_prev -> α_(t−δ)
        # beta_prod_t -> (1 - α_t)
        # beta_prod_t_prev -> (1 - α_(t−δ))
        # sample -> x_t
        # model_output -> e_θ(x_t, t)
        # prev_sample -> x_(t−δ)
        #alpha_prod_t = self.alphas_cumprod[timestep]
        #alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        ac_t = torch.full((b, 1, 1, 1), self.scheduler.alphas_cumprod_t[index], device=device)
        ac_prev = torch.full((b, 1, 1, 1), self.scheduler.alphas_cumprod_prev_t[index], device=device)

        beta_prod_t = 1 - ac_t
        beta_prod_t_prev = 1 - ac_prev

        # corresponds to (α_(t−δ) - α_t) divided by
        # denominator of x_t in formula (9) and plus 1
        # Note: (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
        # sqrt(α_(t−δ)) / sqrt(α_t))
        sample_coeff = (ac_prev / ac_t) ** (0.5)

        # corresponds to denominator of e_θ(x_t, t) in formula (9)
        model_output_denom_coeff = ac_t * beta_prod_t_prev ** (0.5) + (
            ac_t * beta_prod_t * ac_prev
        ) ** (0.5)

        # full formula (9)
        prev_sample = (
            sample_coeff * x - (ac_prev - ac_t) * e_t / model_output_denom_coeff
        )

        return prev_sample, prev_sample

    @torch.no_grad()
    def p_sample_reverse(self, x, c, t, index, **kwargs):
        quantize_denoised = kwargs.get("quantize_denoised", False)
        temperature = kwargs.get("temperature", 1.)
        noise_dropout = kwargs.get("noise_dropout", 0.)
        score_corrector = kwargs.get("score_corrector", None)
        corrector_kwargs = kwargs.get("corrector_kwargs", None)
        uc_scale = kwargs.get("unconditional_guidance_scale", 1.)
        uc = kwargs.get("unconditional_conditioning", None)
        old_eps = kwargs.get("old_eps", [])
        t_next = kwargs.get("t_next", t+1)

        b, *_, device = *x.shape, x.device

        e_t = self._calculate_epsilon(x, c, t, **kwargs)
        e_t, x = self._get_clip_guide(x, t, c, e_t, **kwargs)
            
        ac_next_t = self.safe_to(self.scheduler.alphas_cumprod_t[index])
        ac_t = self.safe_to(self.scheduler.alphas_cumprod_prev_t[index])
        sqrt_one_minus_ac_t = torch.sqrt(1 - ac_t)

        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            #x_prev, pred_x0 = self.get_x_next_and_pred_x0(x, e_t, a_t, a_prev, sigma_t, sqrt_one_minus_at, index, **kwargs)
            x_next, pred_xt = self._get_x_next_and_pred_xt(x, e_t, index)
            
            e_t_next = self._calculate_epsilon(x_next, c, t_next, **kwargs)
            e_t_next, x = self._get_clip_guide(x_next, t_next, c, e_t_next, **kwargs)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        #x_prev, pred_x0 = self.get_x_next_and_pred_x0(x, e_t_prime, a_t, a_prev, sigma_t, sqrt_one_minus_at, index, **kwargs)
        x_next, pred_xt = self._get_x_next_and_pred_xt(x, e_t, index)        

        self.clog("p_sample", f"{t} done")
        return x_next, pred_xt, e_t

            