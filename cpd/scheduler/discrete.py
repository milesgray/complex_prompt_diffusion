import traceback
from typing import Optional, List, Tuple, Union, Any
import math

import torch
import numpy as np

from cpd.util import safe_to, default
from cpd.samplers.extension import create as create_extension
from cpd.scheduler.util import SchedulerOutput

class SigmaScheduler:
    def __init__(self, **kwargs):
        self.algorithm = kwargs.get("sigma_algorithm", "default")
        self.total_steps = kwargs.get("steps", kwargs.get("total_steps", None))
        if self.total_steps:
            self.sigmas = self.get_sigmas(self.algorithm, self.total_steps, **kwargs)
        else:
            self.sigmas = None

    def get_sigmas_karras(self, n, **kwargs):
        """Construct the noise schedule of Karras et al. (2022)."""
        device = kwargs.get('device','cuda')
        sigma_min = kwargs.get("sigma_min",0.1)
        sigma_max = kwargs.get("sigma_max",10)
        rho = kwargs.get("rho",7.)
        ramp = torch.linspace(0, 1, n, 
                              device=device)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def get_sigmas_exponential(self, n, **kwargs):
        """Construct an exponential noise schedule."""
        device = kwargs.get('device','cuda')
        sigma_min=kwargs.get("sigma_min",0.1)
        sigma_max=kwargs.get("sigma_max",10)
        sigmas = torch.linspace(math.log(sigma_max), 
                                math.log(sigma_min), 
                                n, device=device)
        sigmas = sigmas.exp()
        return sigmas

    def get_sigmas_quad(self, n, **kwargs):
        """Construct an quadratic noise schedule."""
        device = kwargs.get('device','cuda')
        sigma_min=kwargs.get("sigma_min",0.1)
        sigma_max=kwargs.get("sigma_max",10)
        sigmas = torch.linspace(math.sqrt(sigma_max), 
                                math.sqrt(sigma_min), 
                                n, device=device)
        sigmas = sigmas ** 2
        return sigmas   

    def get_sigmas_sigmoid(self, n, **kwargs):
        """Construct an sigmoid noise schedule."""
        device = kwargs.get('device','cuda')
        sigma_min=kwargs.get("sigma_min",0.1)
        sigma_max=kwargs.get("sigma_max",10.0)
        sigmas = torch.linspace(-6, 6, n, 
                                device=device)
        sigmas = torch.sigmoid(sigmas) * (sigma_max - sigma_min) * sigma_min
        return sigmas        

    def get_sigmas_vp(self, n, **kwargs):
        """Construct a continuous VP noise schedule."""
        device = kwargs.get('device','cuda')
        beta_d = kwargs.get("beta_d",19.9)
        beta_min = kwargs.get("beta_min",0.1)
        eps_s = kwargs.get("eps_s",1e-3)
        t = torch.linspace(1, eps_s, n, 
                           device=device)
        sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
        return sigmas

    def get_sigmas_linear(self, n, **kwargs):
        """Construct a linear noise schedule."""
        device = kwargs.get('device','cuda')
        if n is None:
            return self.append_zero(self.sigmas.flip(0))
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, 
                           device=device)
        return self.t_to_sigma(t)

    def get_sigmas(self, algorithm, n, **kwargs):
        """
        Construct sigmas according to a specific algorithm.
        
        options: 'linear', 'karras', 'exp', 'quad', 'vp', 'sig'
        """
        if algorithm in ["linear","default"]:
            sigmas = self.get_sigmas_linear(n, **kwargs)
        elif algorithm in ["karras"]:
            sigmas = self.get_sigmas_karras(n, **kwargs)
        elif algorithm in ["exp", "exponential"]:
            sigmas = self.get_sigmas_exponential(n, **kwargs)
        elif algorithm in ["quad", "quadratic"]:
            sigmas = self.get_sigmas_quad(n, **kwargs)
        elif algorithm in ["vp","variance_preserving"]:
            sigmas = self.get_sigmas_vp(n, **kwargs)
        elif algorithm in ["sig", "sigmoid"]:
            sigmas = self.get_sigmas_sigmoid(n, **kwargs)
        else:
            raise NotImplementedError
        self.sigmas =  self.append_zero(sigmas)
        return self.sigmas
    
    def get_scalings(self, x, convert_to_sigma=False):
        if convert_to_sigma:
            sigma = self.t_to_sigma(x)
        else:
            sigma = x
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1 ** 2) ** 0.5
        return c_out, c_in

    
    def sigma_to_t(self, sigma, quantize=None, device="cuda"):
        sigma = sigma.cpu()
        quantize = False #self.quantize if quantize is None else quantize
        dists = torch.abs(sigma - self.sigmas[:, None])
        if quantize:
            return torch.argmin(dists, dim=0).view(sigma.shape)
        low_idx, high_idx = torch.sort(torch.topk(dists, dim=0, k=2, largest=False).indices, dim=0)[0]
        low, high = self.sigmas[low_idx], self.sigmas[high_idx]
        w = (low - sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return safe_to(t.view(sigma.shape), device=device)

    def t_to_sigma(self, t, device="cuda"):
        t = t.cpu().float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        sigma = (1 - w) * self.sigmas[low_idx] + w * self.sigmas[high_idx]
        return safe_to(sigma, device=device)

class BetaScheduler:
    def __init__(self, **kwargs):
        self.schedule = kwargs.get("beta_schedule", "quad")
        self.total_steps = kwargs.get("steps", kwargs.get("total_steps", None))
        if self.total_steps:
            self.betas = self.make_beta_schedule(self.schedule, self.total_steps, **kwargs)
        else:
            self.betas = None
    
    def _warmup_beta(self, linear_start, linear_end, n_timestep, warmup_frac):
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
        warmup_time = int(n_timestep * warmup_frac)
        betas[:warmup_time] = np.linspace(
            linear_start, linear_end, warmup_time, dtype=np.float64)
        return betas
    
    def _betas_for_alpha_bar(self, num_diffusion_timesteps, 
                             max_beta=0.999, 
                             cosine_s=8e-3):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
        (1-beta) over time from t = [0,1].

        Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
        to that part of the diffusion process.


        Args:
            num_diffusion_timesteps (`int`): the number of betas to produce.
            max_beta (`float`): the maximum beta to use; use values lower than 1 to
                        prevent singularities.

        Returns:
            betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
        """

        def alpha_bar(time_step, cosine_s=cosine_s):
            return math.cos((time_step + cosine_s) / (1+cosine_s) * math.pi / 2) ** 2

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas, dtype=np.float64)

    def make_beta_schedule(self, schedule, n_timestep, **kwargs):
        linear_start=kwargs.get("linear_start",1e-6)
        linear_end=kwargs.get("linear_end", 1e-2)
        cosine_s=kwargs.get("cosine_s", 8e-3)
        max_beta=kwargs.get("max_beta", 0.999)
        decimal_precision=kwargs.get("decimal_precision", 4)

        if schedule == 'linear':
            betas = np.linspace(linear_start, linear_end,
                                n_timestep, dtype=np.float64)
        elif schedule == 'quad' or schedule == "scaled_linear":
            betas = np.linspace(linear_start ** 0.5, 
                                linear_end ** 0.5,
                                n_timestep, dtype=np.float64) ** 2
        elif schedule == "exp":
            betas = np.linspace(np.log(linear_start), 
                                np.log(linear_end), 
                                n_timestep, dtype=np.float64)
            betas = np.exp(betas)
        elif schedule == "squaredcos_cap_v2":
            betas = self._betas_for_alpha_bar(n_timestep, 
                                              max_beta=max_beta, 
                                              cosine_s=cosine_s)
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timestep, dtype=np.float64)
            betas = torch.sigmoid(betas) * (linear_end - linear_start) + linear_start
        elif schedule == 'warmup10':
            betas = self._warmup_beta(linear_start, linear_end,
                                      n_timestep, 0.1)
        elif schedule == 'warmup50':
            betas = self._warmup_beta(linear_start, linear_end,
                                      n_timestep, 0.5)
        elif schedule == 'const':
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
        elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = np.linspace(n_timestep, 1, 
                                n_timestep, dtype=np.float64)
            betas = 1 / betas
        elif schedule == "cosine":
            timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
        else:
            raise NotImplementedError(schedule)
        if isinstance(betas, np.ndarray):
            betas = torch.from_numpy(betas)
        betas = betas.clamp(max=max_beta)
        return np.around(betas, decimals=decimal_precision)


class DiscreteBaseScheduler:
    def __init__(self, num_train_timesteps: int = 1000, **kwargs):
        self.total_steps = kwargs.get("steps", kwargs.get("total_steps"))
        self.parameterization = kwargs.get("parameterization", "eps")
        self.verbose = kwargs.get("verbose", False)
        
        self.log = kwargs.get("logger", print)


    def make_timesteps(self, num_timesteps, 
                        discr_method: str ="uniform", 
                        num_train_timesteps: int =1000, 
                        verbose: bool=True,
                        **kwargs):
        if discr_method == 'uniform':
            timesteps = np.asarray(list(range(0, num_train_timesteps, 
                                              num_train_timesteps // num_timesteps)))
        elif discr_method == 'quad':
            timesteps = ((np.linspace(0, np.sqrt(num_train_timesteps * .8), 
                                      num_timesteps)) ** 2).astype(int)
        elif discr_method == 'jumps':                 
            jump_length = kwargs.get("jump_length", 0) # 10 is default in example for inpainting
            jump_n_sample = kwargs.get("jump_n_sample", 0) # 10 is default in example for inpainting

            timesteps = self._add_jumps(num_timesteps, num_train_timesteps,
                                        jump_length=jump_length, 
                                        jump_n_sample=jump_n_sample)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{discr_method}"')

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        steps_out = timesteps + 1
        if steps_out[-1] == self.num_train_timesteps:
            steps_out[-1] = steps_out[-1] - 1
        if verbose:
            self.log(f'Selected timesteps for ddim sampler: {steps_out}')
        return steps_out

    
    def make_negative_prompt_schedule(self, **kwargs):
        schedule = kwargs.get("negative_prompt_schedule", "constant")
        alpha = kwargs.get("negative_prompt_alpha", 1)

        if schedule == 'linear':
            schedule = np.flip(np.linspace(0, 1, self.total_steps))
        elif schedule == 'constant':
            schedule = np.flip(np.ones(self.total_steps))
        elif schedule == 'exp':
            schedule = np.exp(-6 * np.linspace(0, 1, self.total_steps))
        else:
            raise NotImplementedError

        schedule = schedule * alpha

        return schedule
    
    def make_unconditional_prompt_schedule(self, **kwargs):
        schedule = kwargs.get("decaying_uc_schedule", "log")
        uc_scale = kwargs.get("uc_scale", 7.5)
        decay_scale_alpha = kwargs.get("decaying_uc_scale_alpha", 2)
        decay_scale_min = kwargs.get("decaying_uc_scale_min", 2)
        decay_scale_start = kwargs.get("decaying_uc_scale_start", int(self.total_steps * 0.2))
        if schedule == "linear":
            result = np.flip(np.linspace(0,1,self.total_steps))
            result *= uc_scale
            result = np.maximum(result, np.ones_like(result) * decay_scale_min)
        elif schedule == "constant":
            result = np.flip(np.ones(self.total_steps))
            result *= uc_scale
            result = np.maximum(result, np.ones_like(result) * decay_scale_min)
        elif schedule == 'exp':
            result = np.exp(-6 * np.linspace(0, 1, self.total_steps))
            result *= uc_scale
            result = np.maximum(result, np.ones_like(result) * decay_scale_min)
        elif schedule == "log":
            results = []
            for t_idx in range(self.total_steps):
                if decay_scale_start < t_idx:
                    decay_scale_start = min(t_idx, decay_scale_start)       
                    uc_scale = max(
                        decay_scale_min, 
                        uc_scale - (
                            uc_scale * (
                                np.log(t_idx+1-decay_scale_start) /
                                np.log(self.total_steps)
                            )
                        )
                    )
                results.append(uc_scale)
            result = np.array(results)
        else:
            raise NotImplementedError

        return result * decay_scale_alpha
    
    def make_attn_guide_schedule(self, **kwargs):
        schedule = kwargs.get("attn_guide_schedule", "constant")
        alpha = kwargs.get("attn_guide_alpha", 1)

        if schedule == 'linear':
            schedule = np.flip(np.linspace(0, 1, self.total_steps))
        elif schedule == 'constant':
            schedule = np.flip(np.ones(self.total_steps))
        elif schedule == 'exp':
            schedule = np.exp(-6 * np.linspace(0, 1, self.total_steps))
        else:
            raise NotImplementedError

        schedule = schedule * alpha

        return schedule


class DiscreteScheduler(DiscreteBaseScheduler):
    """
    Linear Multistep Scheduler for discrete beta schedules. Based on the original k-diffusion implementation by
    Katherine Crowson:
    https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L181
    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_cosine_s (`float`): TODO
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        verbose (`bool`): Output additional logging
        logger (`callable`): Method that handles log strings
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.parameterization = kwargs.get("parameterization", "eps")
        eta = kwargs.get("eta", 0)
        quantize = kwargs.get("quantize", False)
        
        beta_schedule = kwargs.get("beta_schedule", "quad")
        beta_start = kwargs.get("beta_start", 0.0008)
        beta_end = kwargs.get("beta_end", 0.012)
        beta_max = kwargs.get("beta_max", 0.999)
        beta_cosine_s = kwargs.get("beta_cosine_s", 8e-3)
        v_posterior = kwargs.get("v_posterior", 0.)
        
        verbose = kwargs.get("verbose", None)
        
        self.log = kwargs.get("logger", print)
        self.betas = BetaScheduler().make_beta_schedule(beta_schedule, num_train_timesteps, 
                                             linear_start=beta_start, 
                                             linear_end=beta_end,
                                             cosine_s=beta_cosine_s,
                                             max_beta=beta_max)
        self.alphas = 1. - self.betas.numpy()       
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.final_alpha_cumprod = 1.0
        self.alphas_cumprod_prev = np.append(self.final_alpha_cumprod, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], self.alphas[-1])
        
        assert self.alphas_cumprod.shape[0] == num_train_timesteps, f"alphas have to be defined for each timestep: {self.alphas_cumprod.shape[0]}"
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - v_posterior) * self.betas * (1. - self.alphas_cumprod_prev) / (
                1. - self.alphas_cumprod) + v_posterior * self.betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = safe_to(posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = safe_to(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = safe_to(
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.posterior_mean_coef2 = safe_to(
            (1. - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1. - self.alphas_cumprod))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * safe_to(self.alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(self.alphas_cumprod)) / (2. * 1 - torch.Tensor(self.alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * safe_to(self.alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = lvlb_weights
        assert not torch.isnan(self.lvlb_weights).all()

        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        self.sigmas = safe_to(self.sigmas)
        
        self.one = torch.tensor(1.0)
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0
        
        # setable values
        self.num_inference_steps = None
        self.num_train_timesteps = num_train_timesteps
        self.derivatives = []
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())
        self.quantize = quantize

        self.eta = eta

        if verbose:
            self.log(f'Selected alphas for ddim sampler: a_t: {self.alphas_cumprod_t}; a_(t-1): {self.alphas_cumprod_prev_t}; '
                     f'this results in the following sigma_t schedule for ddim sampler {self.sigmas}')
    
    def set_timesteps(self, num_inference_steps: int, **kwargs):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.
        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        eta = kwargs.get("eta", self.eta)
        verbose = kwargs.get("verbose", False)
        discr_scheduler = kwargs.get("discretize", "uniform")
        
        self.num_inference_steps = num_inference_steps
        self.timesteps = self._make_timesteps(num_inference_steps,
                                              discr_method=discr_scheduler, 
                                              num_train_timesteps=self.num_train_timesteps, 
                                              **kwargs)
        
        self.alphas_cumprod_t = self.alphas_cumprod[self.timesteps]
        self.alphas_cumprod_prev_t = np.asarray([self.alphas_cumprod[0]] + self.alphas_cumprod_t[:-1].tolist())
        self.alphas_cumprod_next_t = np.asarray(self.alphas_cumprod_t[1:].tolist() +  [self.alphas_cumprod[-1]])

        # according the the formula provided in https://arxiv.org/abs/2010.02502
        self.sigmas_t = eta * np.sqrt((1 - self.alphas_cumprod_prev_t) / (1 - self.alphas_cumprod_t) * (1 - self.alphas_cumprod_t / self.alphas_cumprod_prev_t))        
        self.sqrt_one_minus_alphas_cumprod_t = np.sqrt(1. - self.alphas_cumprod_t)

        self.derivatives = []

    def _make_timesteps(self, num_timesteps, 
                        discr_method: str ="uniform", 
                        num_train_timesteps: int =1000, 
                        verbose: bool=True,
                        **kwargs):
        if discr_method == 'uniform':
            timesteps = np.asarray(list(range(0, num_train_timesteps, 
                                              num_train_timesteps // num_timesteps)))
        elif discr_method == 'quad':
            timesteps = ((np.linspace(0, np.sqrt(num_train_timesteps * .8), 
                                      num_timesteps)) ** 2).astype(int)
        elif discr_method == 'jumps':                 
            jump_length = kwargs.get("jump_length", 0) # 10 is default in example for inpainting
            jump_n_sample = kwargs.get("jump_n_sample", 0) # 10 is default in example for inpainting

            timesteps = self._add_jumps(num_timesteps, num_train_timesteps,
                                        jump_length=jump_length, 
                                        jump_n_sample=jump_n_sample)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{discr_method}"')

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        steps_out = timesteps + 1
        if steps_out[-1] == self.num_train_timesteps:
            steps_out[-1] = steps_out[-1] - 1
        if verbose:
            self.log(f'Selected timesteps for ddim sampler: {steps_out}')
        return steps_out

    def _add_jumps(self, num_inference_steps, num_train_timesteps, jump_length=10, jump_n_sample=10):   
        timesteps = []

        jumps = {}
        for j in range(0, num_inference_steps - jump_length, jump_length):
            jumps[j] = jump_n_sample - 1

        t = num_inference_steps
        while t >= 1:
            t = t - 1
            timesteps.append(t)

            if jumps.get(t, 0) > 0:
                jumps[t] = jumps[t] - 1
                for _ in range(jump_length):
                    t = t + 1
                    timesteps.append(t)

        timesteps = np.array(timesteps) * (num_train_timesteps // num_inference_steps)

        return timesteps

    def step(
        self,
        x: Union[torch.FloatTensor, np.ndarray],    # sample
        e_t: Union[torch.FloatTensor, np.ndarray],  # model_output
        t: int,                                     # timestep       
        **kwargs
    ) -> Tuple:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
        Args:
            e_t (`torch.FloatTensor` or `np.ndarray`): direct output from learned diffusion model.
            t (`int`): current discrete timestep in the diffusion chain.
            x (`torch.FloatTensor` or `np.ndarray`):
                current instance of sample being created by diffusion process.
            original_image (`torch.FloatTensor`):
                the original image to inpaint on.
            mask (`torch.FloatTensor`):
                the mask where 0.0 values define which part of the original image to inpaint (change).
            generator (`torch.Generator`, *optional*): random number generator.
        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        deterministic = kwargs.get("deterministic", False)
        temperature = kwargs.get("temperature", 1.)
        noise_dropout = kwargs.get("noise_dropout", 0.)
        clip_sample = kwargs.get("clip_sample", False)
        clip_sample_alg = kwargs.get("clip_sample_alg", "dynamic_thresholding")
        clip_sample_thresh = kwargs.get("clip_sample_thresh", 90)
        verbose = kwargs.get("verbose", False)

        if clip_sample:
            sample_thresholding = create_extension(clip_sample_alg, threshold_x=clip_sample_thresh)
        b, *_, device = *x.shape, x.device
        
        # 1. compute alphas, betas
        a_t = self.expand_value(self.alphas_cumprod_t[t], b, device)
        a_prev = self.expand_value(self.alphas_cumprod_prev_t[t], b, device)
        sqrt_one_minus_at = self.expand_value(self.sqrt_one_minus_alphas_cumprod_t[t], b, device)
        sigma_t = self.expand_value(self.sigmas_t[t], b, device)
        
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # 4. Clip "predicted x_0"
        if clip_sample:
            pred_x0 = sample_thresholding.apply(pred_x0, t, verbose=verbose)
            # the model_output is always re-derived from the clipped x_0 in Glide
            e_t = (x - a_t.sqrt() * pred_x0) / sqrt_one_minus_at

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        try:
            if deterministic:
                noise = torch.ones_like(x)
            else:
                noise = torch.randn(x.shape, device="cpu").to(device)
            noise = sigma_t * noise * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)

            # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        except Exception as e:
            self.log(f"[DiscreteScheduler]\t[step]\t{e}\n{traceback.print_exc()}")
        finally:
            noise = noise.cpu()

        return x_prev, pred_x0
    
    def undo_step(
        self,
        x: Union[torch.FloatTensor, np.ndarray],    # sample
        e_t: Union[torch.FloatTensor, np.ndarray],  # model_output
        t: int,                                     # timestep       
        **kwargs
    ) -> Tuple:
        deterministic = kwargs.get("deterministic", False)
        temperature = kwargs.get("temperature", 1.)
        clip_sample = kwargs.get("clip_sample", False)
        clip_sample_alg = kwargs.get("clip_sample_alg", "dynamic_thresholding")
        clip_sample_thresh = kwargs.get("clip_sample_thresh", 90)
        verbose = kwargs.get("verbose", False)

        if clip_sample:
            sample_thresholding = create_extension(clip_sample_alg, threshold_x=clip_sample_thresh)
        b, *_, device = *x.shape, x.device
        
        # 1. compute alphas, betas
        a_t = self.expand_value(self.alphas_cumprod_t[t], b, device)
        a_prev = self.expand_value(self.alphas_cumprod_prev_t[t], b, device)
        sqrt_one_minus_at = self.expand_value(self.sqrt_one_minus_alphas_cumprod_t[t], b, device)
        sigma_t = self.expand_value(self.sigmas_t[t], b, device)
                
        # UNDO: x = a_prev.sqrt() * pred_x0 + dir_xt + noise
        # x - dir_xt*noise = a_prev.sqrt() * pred_x0
        # (x - dir_xt*noise) / a_prev.sqrt() = pred_x0
        if deterministic:
            noise = torch.ones_like(x)
        else:
            noise = torch.randn(x.shape, device="cpu").to(device)
        noise = sigma_t * noise * temperature
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        pred_x0 = (x - dir_xt*noise) / a_prev.sqrt()

        if clip_sample:
            # UNDO: e_t = (x - a_t.sqrt() * pred_x0) / sqrt_one_minus_at
            # e_t * sqrt_one_minus_at = x - a_t.sqrt() * pred_x0
            # (e_t * sqrt_one_minus_at) + (a_t.sqrt() * pred_x0) = x
            x = (e_t * sqrt_one_minus_at) + (a_t.sqrt() * pred_x0)
        else:
            # UNDO: pred_x0 = (x + sqrt_one_minus_at / e_t) * a_t.sqrt()
            # pred_x0 / a_t.sqrt() = x + sqrt_one_minus_at / e_t
            # (pred_x0 / a_t.sqrt()) - (sqrt_one_minus_at / e_t) = x
            x = (pred_x0 / a_t.sqrt()) - (sqrt_one_minus_at / e_t)
        
        return x, pred_x0

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        t: int,
        noise: torch.FloatTensor=None        
    ) -> torch.FloatTensor:
        """ Stochastic Encode samples.

        Args:
            original_samples (torch.FloatTensor): `x`
            t (int): timestep to target
            noise (torch.FloatTensor, optional): random numbers sampled from normal distribution. 
                Defaults to None.

        Returns:
            torch.FloatTensor: noised sample
        """
        x0 = original_samples
        if noise is None:
            noise = torch.randn_like(x0, device=x0.device)
        a_t = self.alphas_cumprod_t[t]
        sqrt_one_minus_at = self.sqrt_one_minus_alphas_cumprod_t[t]
        
        return (a_t * x0 +
                sqrt_one_minus_at * noise)

    def get_v(self, x, noise, t):
        return (
                self.expand_value(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                self.expand_value(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (self.expand_value(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = self.expand_value(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self.expand_value(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                self.expand_value(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                self.expand_value(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.expand_value(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.expand_value(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (self.expand_value(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                self.expand_value(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    
    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                self.expand_value(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self.expand_value(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
        
    def predict_start_from_z_and_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
                self.expand_value(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                self.expand_value(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                self.expand_value(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                self.expand_value(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def __len__(self):
        return self.num_train_timesteps 

    def expand_value(self, values: Union[np.ndarray, torch.Tensor], b: int, device: str):
        """
        Turns a value into a tensor with 4 dims.
        Args:
            values: value to expand.
            b: batch size
            device: device to create the new tensor on
        Returns:
            a tensor of shape [batch_size, 1, 1, 1] filled with `value`
        """
        values = torch.full((b, 1, 1, 1), values, device=device)

        return values

    
        

    def append_zero(self, x):
        return torch.cat([x, x.new_zeros([1])])