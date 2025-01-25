import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
from functools import partial

from cpd.samplers.diffusion import DiffusionSamplerWrapper
from cpd.samplers.k_diffusion import KDiffusionSampler
from cpd.samplers.registry import register

class AdaptiveDiffusionSampler(KDiffusionSampler):
    def t(self, sigma):
        return -sigma.log()

    def sigma(self, t):
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps, x = self.denoiser.complex_epsilon(x, sigma, *args, **kwargs)
        eps = (x - eps) / self.sigma(t)
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, t, t_next, eps_cache=None, **kwargs):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t, **kwargs)
        x_1 = x - self.sigma(t_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None, **kwargs):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t, **kwargs)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1, **kwargs)
        x_2 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None, **kwargs):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t, **kwargs)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1, **kwargs)
        u2 = x - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2, **kwargs)
        x_3 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return x_3, eps_cache

@register("DPM Fast")
class DPMFastSamplerWrapper(DiffusionSamplerWrapper):
    def __init__(self, name, **kwargs):
        kwargs["constructor"] = DPMFastDiffusionSampler
        super().__init__(name, **kwargs)
class DPMFastDiffusionSampler(AdaptiveDiffusionSampler):
    def __init__(self, model):
        super().__init__(model, "dpm fast")
    
    @torch.no_grad()
    def _sampling(self, x, sigmas, model_args=None, **kwargs):
        callback = kwargs.get("callback", None) 
        disable = kwargs.get("disable", None) 
        eta = kwargs.get("eta", 0.)
        s_noise = kwargs.get("s_noise", 1.)
        
        t_start = self.t(torch.tensor(sigmas[0]))
        t_end = self.t(torch.tensor(sigmas[-2]))
        nfe = len(sigmas) - 1
        m = math.floor(nfe / 3) + 1
        ts = torch.linspace(t_start, t_end, m + 1, device=x.device)

        if nfe % 3 == 0:
            orders = [3] * (m - 2) + [2, 1]
        else:
            orders = [3] * (m - 1) + [nfe % 3]

        for i in trange(len(orders), disable=disable):
            eps_cache = {}
            t, t_next = ts[i], ts[i + 1]
            gamma = eta * torch.sqrt(2 * (t_next - t))
            t = torch.maximum(t_start, t - gamma.log1p())
            noise = torch.randn_like(x) * s_noise
            if t < ts[i]:
                x = x + noise * (self.sigma(t) ** 2 - self.sigma(ts[i]) ** 2).sqrt()

            eps, eps_cache = self.eps(eps_cache, 'eps', x, t, **kwargs)
            denoised = x - self.sigma(t) * eps
                
            if callback is not None:
                callback({'x': x, 'i': i, 't': ts[i], 't_up': t, 'denoised': denoised})

            if orders[i] == 1:
                x, eps_cache = self.dpm_solver_1_step(x, t, t_next, eps_cache=eps_cache, **kwargs)
            elif orders[i] == 2:
                x, eps_cache = self.dpm_solver_2_step(x, t, t_next, eps_cache=eps_cache, **kwargs)
            else:
                x, eps_cache = self.dpm_solver_3_step(x, t, t_next, eps_cache=eps_cache, **kwargs)

        return x


@register("DPM Adaptive")
class DPMAdaptiveSamplerWrapper(DiffusionSamplerWrapper):
    def __init__(self, name, **kwargs):
        kwargs["constructor"] = DPM2AdaptiveDiffusionSampler
        super().__init__(name, **kwargs)
class DPM2AdaptiveDiffusionSampler(AdaptiveDiffusionSampler):
    """Ancestral sampling with DPM-Solver inspired second-order steps."""
    
    def __init__(self, model):
        super().__init__(model, "dpm adaptive")
    
    @torch.no_grad()
    def _sampling(self, x, sigmas, model_args=None, **kwargs):
        callback = kwargs.get("callback", None) 
        disable = kwargs.get("disable", None)
        
        t_start = self.t(torch.tensor(sigmas[0]))
        t_end = self.t(torch.tensor(sigmas[-2]))
        order=3
        rtol=0.05
        atol=0.0078
        h_init=0.05
        pcoeff=0.
        icoeff=1.
        dcoeff=0.
        accept_safety=0.81,

        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        x_prev = x
        pid = PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, 
                                    order=order, 
                                    accept_safety=accept_safety)
        info = {'steps': 0, 'nfe': 0, 'n_accept': 0, 'n_reject': 0}

        while s < t_end - 1e-5:
            eps_cache = {}

            eps, eps_cache = self.eps(eps_cache, 'eps', x, s, **kwargs)
            denoised = x - self.sigma(s) * eps

            t = torch.minimum(t_end, s + pid.h)
            if order == 2:
                x_low, eps_cache = self.dpm_solver_1_step(x, s, t, eps_cache=eps_cache, **kwargs)
                x_high, eps_cache = self.dpm_solver_2_step(x, s, t, eps_cache=eps_cache, **kwargs)
            else:
                x_low, eps_cache = self.dpm_solver_2_step(x, s, t, r1=1 / 3, eps_cache=eps_cache, **kwargs)
                x_high, eps_cache = self.dpm_solver_3_step(x, s, t, eps_cache=eps_cache, **kwargs)
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) * x.numel() ** -(1 / 2)
            if pid.propose_step(error):
                x_prev = x_low
                x = x_high
                s = t
                info['n_accept'] += 1
            else:
                info['n_reject'] += 1
            info['nfe'] += order
            info['steps'] += 1

            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': info['steps'] - 1, 't': s, 't_up': s, 'denoised': denoised, 'error': error, 'h': pid.h, **info})

        return x

class PIDStepSizeController:
    """A PID controller for ODE adaptive step size control."""
    def __init__(self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8):
        self.h = h
        self.b1 = (pcoeff + icoeff + dcoeff) / order
        self.b2 = -(pcoeff + 2 * dcoeff) / order
        self.b3 = dcoeff / order
        self.accept_safety = accept_safety
        self.eps = eps
        self.errs = []

    def limiter(self, x):
        return 1 + math.atan(x - 1)

    def propose_step(self, error):
        inv_error = 1 / (float(error) + self.eps)
        if not self.errs:
            self.errs = [inv_error, inv_error, inv_error]
        self.errs[0] = inv_error
        factor = self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
        factor = self.limiter(factor)
        accept = factor >= self.accept_safety
        if accept:
            self.errs[2] = self.errs[1]
            self.errs[1] = self.errs[0]
        self.h *= factor
        return accept


class DPMSolver(nn.Module):
    """DPM-Solver. See https://arxiv.org/abs/2206.00927."""

    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma):
        return -sigma.log()

    def sigma(self, t):
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (x - self.model(x, sigma, *args, **self.extra_args, **kwargs)) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, t, t_next, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        x_1 = x - self.sigma(t_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        x_2 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        u2 = x - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2)
        x_3 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return x_3, eps_cache

    def dpm_solver_fast(self, x, t_start, t_end, nfe, eta=0., s_noise=1.):
        m = math.floor(nfe / 3) + 1
        ts = torch.linspace(t_start, t_end, m + 1, device=x.device)

        if nfe % 3 == 0:
            orders = [3] * (m - 2) + [2, 1]
        else:
            orders = [3] * (m - 1) + [nfe % 3]

        for i in range(len(orders)):
            eps_cache = {}
            t, t_next = ts[i], ts[i + 1]
            gamma = eta * torch.sqrt(2 * (t_next - t))
            t = torch.maximum(t_start, t - gamma.log1p())
            noise = torch.randn_like(x) * s_noise
            if t < ts[i]:
                x = x + noise * (self.sigma(t) ** 2 - self.sigma(ts[i]) ** 2).sqrt()

            eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
            denoised = x - self.sigma(t) * eps
            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': i, 't': ts[i], 't_up': t, 'denoised': denoised})

            if orders[i] == 1:
                x, eps_cache = self.dpm_solver_1_step(x, t, t_next, eps_cache=eps_cache)
            elif orders[i] == 2:
                x, eps_cache = self.dpm_solver_2_step(x, t, t_next, eps_cache=eps_cache)
            else:
                x, eps_cache = self.dpm_solver_3_step(x, t, t_next, eps_cache=eps_cache)

        return x

    def dpm_solver_adaptive(self, x, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81):
        assert order in {2, 3}
        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        x_prev = x
        pid = PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, order, accept_safety)
        info = {'steps': 0, 'nfe': 0, 'n_accept': 0, 'n_reject': 0}

        while s < t_end - 1e-5:
            eps_cache = {}

            eps, eps_cache = self.eps(eps_cache, 'eps', x, s)
            denoised = x - self.sigma(s) * eps

            t = torch.minimum(t_end, s + pid.h)
            if order == 2:
                x_low, eps_cache = self.dpm_solver_1_step(x, s, t, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_2_step(x, s, t, eps_cache=eps_cache)
            else:
                x_low, eps_cache = self.dpm_solver_2_step(x, s, t, r1=1 / 3, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_3_step(x, s, t, eps_cache=eps_cache)
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) * x.numel() ** -(1 / 2)
            if pid.propose_step(error):
                x_prev = x_low
                x = x_high
                s = t
                info['n_accept'] += 1
            else:
                info['n_reject'] += 1
            info['nfe'] += order
            info['steps'] += 1

            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': info['steps'] - 1, 't': s, 't_up': s, 'denoised': denoised, 'error': error, 'h': pid.h, **info})

        return x, info


@torch.no_grad()
def sample_dpm_fast(model, x, sigma_min, sigma_max, n, extra_args=None, callback=None, disable=None, eta=0., s_noise=1.):
    """DPM-Solver-Fast (fixed step size). See https://arxiv.org/abs/2206.00927."""
    with tqdm(total=n, disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        return dpm_solver.dpm_solver_fast(x, 
        dpm_solver.t(torch.tensor(sigma_max)), 
        dpm_solver.t(torch.tensor(sigma_min)), 
        n, 
        eta, 
        s_noise)


@torch.no_grad()
def sample_dpm_adaptive(model, x, sigma_min, sigma_max, extra_args=None, callback=None, disable=None, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, return_info=False):
    """DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927."""
    with tqdm(disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        x, info = dpm_solver.dpm_solver_adaptive(x, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety)
    if return_info:
        return x, info
    return x

def dynamic_clip(x, threshold=99.5):
    s = np.percentile(np.abs(x.cpu()), threshold, axis=tuple(range(1,x.ndim)))
    s = np.max(np.append(s,1.0))
    return torch.div(x, s)

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])