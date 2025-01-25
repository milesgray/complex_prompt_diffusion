from typing import Optional, List, Tuple, Union, Any

import torch
import numpy as np

from cpd.scheduler.util import SchedulerOutput

class LMSDiscreteScheduler:
    """
    Linear Multistep Scheduler for discrete beta schedules. Based on the original k-diffusion implementation by
    Katherine Crowson:
    https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L181
    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, optional): TODO
            options to clip the variance used when adding noise to the denoised sample. Choose from `fixed_small`,
            `fixed_small_log`, `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        timestep_values (`np.ndarry`, optional): TODO
        tensor_format (`str`): whether the scheduler expects pytorch or numpy arrays.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        timestep_values: Optional[np.ndarray] = None,
        tensor_format: str = "pt",
    ):
        if trained_betas is not None:
            self.betas = np.asarray(trained_betas)
        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32) ** 2
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        self.num_train_timesteps= num_train_timesteps
        self.derivatives = []

    def get_lms_coefficient(self, order, t, current_order):
        """
        Compute a linear multistep coefficient.
        Args:
            order (TODO):
            t (TODO):
            current_order (TODO):
        """

        def lms_derivative(tau):
            prod = 1.0
            for k in range(order):
                if current_order == k:
                    continue
                prod *= (tau - self.sigmas[t - k]) / (self.sigmas[t - current_order] - self.sigmas[t - k])
            return prod

        integrated_coeff = integrate.quad(lms_derivative, self.sigmas[t], self.sigmas[t + 1], epsrel=1e-4)[0]

        return integrated_coeff

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.
        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=float)

        low_idx = np.floor(self.timesteps).astype(int)
        high_idx = np.ceil(self.timesteps).astype(int)
        frac = np.mod(self.timesteps, 1.0)
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = (1 - frac) * sigmas[low_idx] + frac * sigmas[high_idx]
        self.sigmas = np.concatenate([sigmas, [0.0]])

        self.derivatives = []

    def step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
        order: int = 4,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
        Args:
            model_output (`torch.FloatTensor` or `np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor` or `np.ndarray`):
                current instance of sample being created by diffusion process.
            order: coefficient for multi-step inference.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class
        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        sigma = self.sigmas[timestep]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        pred_original_sample = sample - sigma * model_output

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma
        self.derivatives.append(derivative)
        if len(self.derivatives) > order:
            self.derivatives.pop(0)

        # 3. Compute linear multistep coefficients
        order = min(timestep + 1, order)
        lms_coeffs = [self.get_lms_coefficient(order, timestep, curr_order) for curr_order in range(order)]

        # 4. Compute previous sample based on the derivatives path
        prev_sample = sample + sum(
            coeff * derivative for coeff, derivative in zip(lms_coeffs, reversed(self.derivatives))
        )

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: Union[torch.FloatTensor, np.ndarray],
        noise: Union[torch.FloatTensor, np.ndarray],
        timesteps: Union[torch.IntTensor, np.ndarray],
    ) -> Union[torch.FloatTensor, np.ndarray]:
        sigmas = self.match_shape(self.sigmas[timesteps], noise)
        noisy_samples = original_samples + noise * sigmas

        return noisy_samples


    def match_shape(self, values: Union[np.ndarray, torch.Tensor], broadcast_array: Union[np.ndarray, torch.Tensor]):
        """
        Turns a 1-D array into an array or tensor with len(broadcast_array.shape) dims.
        Args:
            values: an array or tensor of values to extract.
            broadcast_array: an array with a larger shape of K dimensions with the batch
                dimension equal to the length of timesteps.
        Returns:
            a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """

        tensor_format = getattr(self, "tensor_format", "pt")
        values = values.flatten()

        while len(values.shape) < len(broadcast_array.shape):
            values = values[..., None]

        return values

    def __len__(self):
        return self.config.num_train_timesteps 