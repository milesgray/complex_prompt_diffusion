from typing import Union, Tuple
import math
import numpy as np
import torch

from cpd.scheduler.discrete import DiscreteScheduler
from cpd.scheduler.util import SchedulerOutput

class IPNDMScheduler(DiscreteScheduler):
    """
    Improved Pseudo numerical methods for diffusion models (iPNDM) ported from @crowsonkb's amazing k-diffusion
    [library](https://github.com/crowsonkb/v-diffusion-pytorch/blob/987f8985e38208345c1959b0ea767a625831cc9b/diffusion/sampling.py#L296)
    """

    def set_timesteps(self, steps: int, **kwargs):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.
        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        device = kwargs.get("device", "cuda")
        self.num_inference_steps = steps
        steps = torch.linspace(1, 0, steps + 1)[:-1]
        steps = torch.cat([steps, torch.tensor([0.0])])

        self.betas = torch.sin(steps * math.pi / 2) ** 2
        self.alphas = (1.0 - self.betas**2) ** 0.5

        timesteps = (torch.atan2(self.betas, self.alphas) / math.pi * 2)[:-1]
        self.timesteps = timesteps.to(device)

        self.ets = []

    def step(
        self,
        x: Union[torch.FloatTensor, np.ndarray],    # sample
        e_t: Union[torch.FloatTensor, np.ndarray],  # model_output
        t: int,                                     # timestep       
        **kwargs
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the linear multi-step method. This has one forward pass with multiple
        times to approximate the solution.
        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class
        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        timestep_index = (self.timesteps == t).nonzero().item()
        prev_timestep_index = timestep_index + 1

        ets = x * self.betas[timestep_index] + e_t * self.alphas[timestep_index]
        self.ets.append(ets)

        if len(self.ets) == 1:
            ets = self.ets[-1]
        elif len(self.ets) == 2:
            ets = (3 * self.ets[-1] - self.ets[-2]) / 2
        elif len(self.ets) == 3:
            ets = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12
        else:
            ets = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])

        prev_sample, pred = self._get_prev_sample(x, timestep_index, prev_timestep_index, ets)

        return prev_sample, pred

    def _get_prev_sample(self, sample, timestep_index, prev_timestep_index, ets):
        alpha = self.alphas[timestep_index]
        sigma = self.betas[timestep_index]

        next_alpha = self.alphas[prev_timestep_index]
        next_sigma = self.betas[prev_timestep_index]

        pred = (sample - sigma * ets) / max(alpha, 1e-8)
        prev_sample = next_alpha * pred + ets * next_sigma

        return prev_sample, pred

    def undo_step(
        self,
        x: Union[torch.FloatTensor, np.ndarray],    # sample
        e_t: Union[torch.FloatTensor, np.ndarray],  # model_output
        t: int,                                     # timestep       
        **kwargs
    ) -> Tuple:
        timestep_index = (self.timesteps == t).nonzero().item()
        next_timestep_index = timestep_index - 1
        sample, ets = self._get_next_sample(x, timestep_index, next_timestep_index)

        if len(self.ets) > 3:
            ets = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])
        elif len(self.ets) == 3:
            ets = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12

        elif len(self.ets) == 2:
            ets = (3 * self.ets[-1] - self.ets[-2]) / 2
        elif len(self.ets) == 1:
            ets = self.ets[-1]

        next_sample, ets = self._get_next_sample(x, timestep_index, next_timestep_index, ets)

        return next_sample, ets


    def _get_next_sample(self, prev_sample, timestep_index, next_timestep_index, pred):
        alpha = self.alphas[timestep_index]
        sigma = self.betas[timestep_index]

        prev_alpha = self.alphas[next_timestep_index]
        prev_sigma = self.betas[next_timestep_index]

        #pred = (sample - sigma * ets) / max(alpha, 1e-8)
        #prev_sample = next_alpha * pred + ets * next_sigma
        ets = (prev_sample - ets * sigma) / alpha 
        sample = (pred * max(prev_alpha, 1e-8)) + (prev_sigma * ets)

        return sample, ets