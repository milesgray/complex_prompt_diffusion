import traceback
from typing import Optional, List, Tuple, Union, Any

import torch
import numpy as np

from cpd.samplers.extension import create as create_extension
from cpd.scheduler.util import SchedulerOutput
from cpd.scheduler.ddim import DiscreteScheduler

class RePaintScheduler(DiscreteScheduler):
    """
    RePaint is a schedule for DDPM inpainting inside a given mask.

    For more details, see the original paper: https://arxiv.org/pdf/2201.09865.pdf

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        eta (`float`):
            The weight of noise for added noise in a diffusion step. Its value is between 0.0 and 1.0 -0.0 is DDIM and
            1.0 is DDPM scheduler respectively.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        variance_type (`str`):
            options to clip the variance used when adding noise to the denoised sample. Choose from `fixed_small`,
            `fixed_small_log`, `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        clip_sample (`bool`, default `True`):
            option to clip predicted sample between -1 and 1 for numerical stability.

        verbose (`bool`): Output additional logging
        logger (`callable`): Method that handles log strings
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        **kwargs
    ):
        eta = kwargs.get("eta", 0)
        beta_schedule = kwargs.get("beta_schedule", "linear")
        beta_start = kwargs.get("beta_start", 0.0008)
        beta_end = kwargs.get("beta_end", 0.012)
        beta_max = kwargs.get("beta_max", 0.999)
        beta_cosine_s = kwargs.get("beta_cosine_s", 8e-3)
        verbose = kwargs.get("verbose", None)
        
        self.log = kwargs.get("logger", print)
        self.betas = self._make_beta_schedule(beta_schedule, num_train_timesteps, 
                                             linear_start=beta_start, 
                                             linear_end=beta_end,
                                             cosine_s=beta_cosine_s,
                                             max_beta=beta_max)
        self.alphas = 1. - self.betas.numpy()       
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.final_alpha_cumprod = 1.0
        self.alphas_cumprod_prev = np.append(self.final_alpha_cumprod, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], self.alphas[-1])
        
        assert self.alphas_cumprod.shape[0] == num_train_timesteps, f"alphas have to be defined for each timestep: {alphas_cumprod.shape[0]}"
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)
        
        self.one = torch.tensor(1.0)
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0
        
        # setable values
        self.num_inference_steps = None
        self.num_train_timesteps = num_train_timesteps
        self.derivatives = []
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        self.eta = eta

        if verbose:
            self.log(f'Selected alphas for ddim sampler: a_t: {self.alphas_cumprod_t}; a_(t-1): {self.alphas_cumprod_prev_t}; '
                     f'this results in the following sigma_t schedule for ddim sampler {self.sigmas}')

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

    def _make_timesteps(self, num_timesteps, 
                        jump_length: int=10,
                        jump_n_sample: int=10,
                        discr_method: str ="jumps", 
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

    def set_timesteps(
        self, 
        num_inference_steps: int, 
        jump_length: int = 10,
        jump_n_sample: int = 10,
        **kwargs
    ):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.
        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        eta = kwargs.get("eta", self.eta)
        verbose = kwargs.get("verbose", False)
        discr_scheduler = kwargs.get("discretize", "jumps")
        
        self.num_inference_steps = num_inference_steps
        self.timesteps = self._make_timesteps(num_inference_steps,
                                              discr_method=discr_scheduler, 
                                              num_train_timesteps=self.num_train_timesteps, 
                                              verbose=verbose)
        
        self.alphas_cumprod_t = self.alphas_cumprod[self.timesteps]
        self.alphas_cumprod_prev_t = np.asarray([self.alphas_cumprod[0]] + self.alphas_cumprod[self.timesteps[:-1]].tolist())
        self.alphas_cumprod_next_t = np.asarray(self.alphas_cumprod[1:].tolist() +  [self.alphas[-1]])

        # according the the formula provided in https://arxiv.org/abs/2010.02502
        self.sigmas_t = eta * np.sqrt((1 - self.alphas_cumprod_prev_t) / (1 - self.alphas_cumprod_t) * (1 - self.alphas_cumprod_t / self.alphas_cumprod_prev_t))        
        self.sqrt_one_minus_alphas_cumprod_t = np.sqrt(1. - self.alphas_cumprod_t)

        self.derivatives = []

    def _get_variance(self, t, noise):
        prev_timestep = t - self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[prev_timestep]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from
        # https://arxiv.org/pdf/2006.11239.pdf) and sample from it to get
        # previous sample x_{t-1} ~ N(pred_prev_sample, variance) == add
        # variance to pred_sample
        # Is equivalent to formula (16) in https://arxiv.org/pdf/2010.02502.pdf
        # without eta.
        # variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def step(
        self,
        x: Union[torch.FloatTensor, np.ndarray],    # sample
        e_t: Union[torch.FloatTensor, np.ndarray],  # model_output
        t: int,                                     # timestep
        original_image: torch.FloatTensor,
        mask: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
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
        temperature = kwargs.get("temperature", 1.)
        noise_dropout = kwargs.get("noise_dropout", 0.)
        clip_sample = kwargs.get("clip_sample", False)
        clip_sample_alg = kwargs.get("clip_sample_alg", "dynamic_thresholding")
        clip_sample_thresh = kwargs.get("clip_sample_thresh", 90)
        verbose = kwargs.get("verbose", False)

        if clip_sample:
            sample_thresholding = create_extension(clip_sample_alg, threshold_x=clip_sample_thresh)
        b, *_, device = *x.shape, x.device
        
        try:
            # 1. compute alphas, betas
            alpha_prod_t = self.expand_value(self.alphas_cumprod_t[t], b, device)
            alpha_prod_t_prev = self.expand_value(self.alphas_cumprod_prev_t[t], b, device)
            beta_prod_t = self.expand_value(self.sqrt_one_minus_alphas_cumprod_t[t], b, device)
            sigma_t = self.expand_value(self.sigmas_t[t], b, device)

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5

            # 3. Clip "predicted x_0"
            if clip_sample:
                pred_original_sample = sample_thresholding.apply(pred_original_sample, t, verbose=verbose)
                # the model_output is always re-derived from the clipped x_0 in Glide
                e_t = (x - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t

            # We choose to follow RePaint Algorithm 1 to get x_{t-1}, however we
            # substitute formula (7) in the algorithm coming from DDPM paper
            # (formula (4) Algorithm 2 - Sampling) with formula (12) from DDIM paper.
            # DDIM schedule gives the same results as DDPM with eta = 1.0
            # Noise is being reused in 7. and 8., but no impact on quality has
            # been observed.

            # 5. Add noise
            noise = torch.randn(
                e_t.shape, dtype=e_t.dtype, generator=generator, device=e_t.device
            ) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)

            std_dev_t = self.eta * self._get_variance(timestep) ** 0.5

            variance = 0
            if t > 0 and self.eta > 0:
                variance = std_dev_t * noise
                
            # 6. compute "direction pointing to x_t" of formula (12)
            # from https://arxiv.org/pdf/2010.02502.pdf
            pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * e_t

            # 7. compute x_{t-1} of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            prev_unknown_part = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction + variance

            # 8. Algorithm 1 Line 5 https://arxiv.org/pdf/2201.09865.pdf
            prev_known_part = (alpha_prod_t**0.5) * original_image + ((1 - alpha_prod_t) ** 0.5) * noise

            # 9. Algorithm 1 Line 8 https://arxiv.org/pdf/2201.09865.pdf
            pred_prev_sample = mask * prev_known_part + (1.0 - mask) * prev_unknown_part
        except Exception as e:
            self.log(f"[RepaintScheduler]\t[step]\t{e}\n{traceback.print_exc()}")
        finally:
            noise = noise.cpu()

        return x_prev, pred_x0

    def undo_step(self, sample, timestep, generator=None):
        n = self.config.num_train_timesteps // self.num_inference_steps

        for i in range(n):
            beta = self.betas[timestep + i]
            noise = torch.randn(sample.shape, generator=generator, device=sample.device)

            # 10. Algorithm 1 Line 10 https://arxiv.org/pdf/2201.09865.pdf
            sample = (1 - beta) ** 0.5 * sample + beta**0.5 * noise

        return sample
        
    def __len__(self):
        return self.num_train_timesteps 