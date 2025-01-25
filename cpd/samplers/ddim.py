""" 
## DDIM Sampler
Based on CompVis implementation, but with only sampling related functionality refactored
and pimped out with additional tricks from various papers.

- Used as a base class for the PLMS Sampler.
"""
import math
import traceback
from typing import Union, Optional
from functools import partial

import torch
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import PIL

from cpd.util import CudaMon, safe_to
from cpd.samplers.diffusion import DiffusionSamplerWrapper, DiffusionSampler
from cpd.samplers.registry import register
from cpd.samplers.extension.callbacks import render_callback
from cpd.scheduler.discrete import DiscreteScheduler
from cpd.scheduler.multistep import DPMSolverMultistepScheduler
from cpd.scheduler.repaint import RePaintScheduler

@register("DDIM")
class DDIMSamplerWrapper(DiffusionSamplerWrapper):
    def __init__(self, name, **kwargs):
        kwargs["constructor"] = DDIMSampler
        super().__init__(name, **kwargs)

class DDIMSampler(DiffusionSampler):
    def __init__(self, model, **kwargs): 
        super().__init__(model, **kwargs)       
        self.log = kwargs.get("logger", print)
        self.name = kwargs.get("name", "DDIM")
        
        self.ddpm_num_timesteps = 1000
        scheduler = kwargs.get("schedulerClass", "discrete")
        if scheduler == "discrete":
            self.scheduler = DiscreteScheduler(
                num_train_timesteps=self.ddpm_num_timesteps, **kwargs
            )
        elif scheduler == "multistep":
            self.scheduler = DPMSolverMultistepScheduler(num_train_timesteps=self.ddpm_num_timesteps, **kwargs)
        elif scheduler == "repaint":
            self.scheduler = RePaintScheduler(num_train_timesteps=self.ddpm_num_timesteps, **kwargs)
        
        # For the intermediary decoded output used as input to the VAE when
        # generating gradients for CLIP guidance
        if "feature_extractor" in model:
            # The feature extractor happens to have the normalization params that the VAE expects
            self.feature_extractor = model["feature_extractor"]
            self.normalize = transforms.Normalize(
                mean=self.feature_extractor.image_mean,
                std=self.feature_extractor.image_std,
            )
        else:
            # these are the values from the feature extractor supplied by CompVis
            # mean: [0.48145466, 0.4578275, 0.40821073]
            image_mean = [0.48145466, 0.4578275, 0.40821073]
            # std: [0.26862954, 0.26130258, 0.27577711]
            image_std = [0.26862954, 0.26130258, 0.27577711]
            self.normalize = transforms.Normalize(mean=image_mean, std=image_std)

        self.blur = transforms.GaussianBlur(kernel_size=31)
        self.clog = CudaMon("ddim sampler")
        self.clog("__init__", "done init")

    @torch.no_grad()
    def sample(self, steps, batch_size, shape, **kwargs):
        """Generate image with reverse diffusion using unet model.

        Args:
            steps (int): Number of diffusion steps
            batch_size (int): Number of concurrent outputs to generate
            shape (tuple): Channels, Height, Width

        Returns:
            tuple(torch.Tensor, dict): Output images, intermediary steps
        """        
        conditioning = kwargs.get("conditioning", None)
        verbose = kwargs.get("verbose", None)
        decode = kwargs.get("decode", False)

        self.scheduler.set_timesteps(steps, **kwargs)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        if verbose:
            self.log(f"[{self.name}]\tData shape for sampling is {size}")

        if decode:
            assert (
                "denoising_strength" in kwargs and "x_T" in kwargs
            ), "Must include 'x_T' (noise to decode) and 'denoising_strength' as keyword args"
            denoising_strength = kwargs.get("denoising_strength")
            t_start = int((1 - denoising_strength) * len(self.scheduler.timesteps))
            timesteps = self.scheduler.timesteps[:t_start]            
            kwargs["x_T"] = self.scheduler.add_noise(kwargs.get("x_T"), t_start)
        else:
            timesteps = self.scheduler.timesteps

        kwargs["total_steps"] = len(timesteps)
        samples, intermediates = self._sampling(conditioning, size, timesteps, **kwargs)
        self.clog("sample", "done")
        return samples, intermediates

    @torch.no_grad()
    def _sampling(self, cond, shape, timesteps, **kwargs):
        x_T = kwargs.get("x_T", None)
        x0 = kwargs.get("x0", None)
        callback = kwargs.get("callback", None)
        img_callback = kwargs.get("img_callback", None)
        verbose = kwargs.get("verbose", False)
        silent = kwargs.get("silent", False)

        device = self.device
        b = shape[0]
        if x_T is None:
            x_T = torch.randn(shape, device=device)

        time_range = np.flip(timesteps)
        if verbose:
            self.log(f"[{self.name}]\t Running Sampling with {timesteps.shape[0]} timesteps")
        iterator = tqdm(
            time_range, desc=f"{self.name} Sampler", total=timesteps.shape[0]
        )
        kwargs["bar"] = iterator

        old_eps = []
        old_preds = []
        for i, step in enumerate(iterator):
            index = timesteps.shape[0] - i - 1
            kwargs["t_idx"] = i
            try:
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                ts_next = torch.full(
                    (b,),
                    time_range[min(i + 1, len(time_range) - 1)],
                    device=device,
                    dtype=torch.long,
                )

                kwargs["old_eps"] = old_eps
                kwargs["t_next"] = ts_next

                outs = self.p_sample_reverse(x_T, cond, ts, index, **kwargs)
            except Exception as e:
                self.log(
                    f"[{self.name}]\t[_sampling]\terror\t{e}\n{traceback.print_exc()}"
                )
            finally:
                ts = ts.cpu()
                ts_next = ts_next.cpu()
            x_T, pred_x0, e_t = outs

            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)
            render_callback(pred_x0, i, **kwargs)
        img = x_T
        self.clog("_sampling", "done")
        return img, []

    def _log_calculate_epsilon(self, e_t, e_t_prime, t, bar=None):
        diff_large = e_t.gt(e_t_prime).float().mean().item()
        diff_mean = (e_t.mean() - e_t_prime.mean()).item()
        diff_std = (e_t.std() - e_t_prime.std()).item()
        diff_dist = (e_t - e_t_prime).mean().item()
        if bar is None:
            self.log(
                f"[{t.item()}]\tlarge: {diff_large:0.5f}\tmean: {diff_mean:0.5f}\tstd: {diff_std:0.5f}\tdist: {diff_dist:0.5f}"
            )
        else:
            stats = {
                "large": f"{diff_large:0.5f}",
                "mean": f"{diff_mean:0.5f}",
                "std": f"{diff_std:0.5f}",
                "dist": f"{diff_dist:0.5f}",
            }
            bar.set_postfix(stats)

    def _attn_guidance(self, x_in, t_in, f_uc, **kwargs):
        t_idx = kwargs.get("t_idx")
        x = x_in[0]
        safe_to_x = partial(safe_to, device=x.device, dtype=x.dtype)

        attn_guide_mode = kwargs.get("attn_guide_mode", 2)
        attn_guide_idx = kwargs.get("attn_guide_idx",kwargs.get("return_attn_idx", -1))        
        
        attn_guide_mask_idx = kwargs.get("attn_guide_mask_idx", 0)
        attn_guide_mask_threshold = kwargs.get("attn_guide_mask_threshold", kwargs.get("attn_mask_threshold", 90))
        attn_guide_blur_k = kwargs.get("attn_guide_blur_k", 31)

        self.blur = transforms.GaussianBlur(kernel_size=attn_guide_blur_k)
        if attn_guide_mode != 2:            
            ac_t = safe_to_x(self.scheduler.alphas_cumprod_t[t_idx])
            sqrt_one_minus_act = safe_to_x(self.scheduler.sqrt_one_minus_alphas_cumprod_t[t_idx])

        out = torch.zeros_like(t_in)
        attn_out = [0]
        t = t_in[0]
        uc = f_uc[0]
        try:
            out, attn = self.unet(x_in, t_in, f_uc, return_attn=True)
            if hasattr(out, "sample"):
                out = out.sample
            mask = attn[attn_guide_idx].mean(1, keepdims=True)                        
            s = np.percentile(mask.detach().cpu(), 
                              attn_guide_mask_threshold, 
                              axis=tuple(range(0,mask.ndim)))
            mask[mask > s] = 1
            mask[mask < s] = 0
            mask = mask[attn_guide_mask_idx]
            
            if attn_guide_mode == 2:
                cwargs = kwargs.copy()
                cwargs["deterministic"] = True
                cwargs["clip_sample"] = False
                sample, _ = self.scheduler.step(x, out[0], t_idx, **cwargs)                    
            else:
                sample = (x - sqrt_one_minus_act * out[0]) / ac_t.sqrt()

            blur_sample = self.blur(sample)
            
            if attn_guide_mode == 2:
                blur_x, _ = self.scheduler.undo_step(blur_sample, out[0], t_idx, **cwargs)
            elif attn_guide_mode == 1:
                blur_x = (blur_sample * ac_t.sqrt()) + (sqrt_one_minus_act * out[0])
            else:
                blur_x = ((blur_sample + sqrt_one_minus_act) / out[0]) * ac_t.sqrt()

            masked_x = blur_x * mask
            guide_x = masked_x + (x * (1 - mask))
            attn_out = self.unet(guide_x[0].unsqueeze(0), t.unsqueeze(0), uc.unsqueeze(0).cuda())
            if hasattr(attn_out, "sample"):
                attn_out = attn_out.sample
        except Exception as e:
            self.log(f"[{self.name}]\t[_attn_guidance]\t{e}\n{traceback.print_exc()}")
        finally:
            guide_x = guide_x.cpu()          
        return out, attn_out[0]
            
    @torch.no_grad()
    def _calculate_epsilon(self, x, c, t, **kwargs):
        verbose = kwargs.get("verbose", False)        
        bar = kwargs.get("bar", None)
        # Time
        t_idx = kwargs.get("t_idx", 0)
        total_steps = kwargs.get("total_steps", 1000)
        
        # Unconditional Conditioning Scale
        uc = kwargs.get("unconditional_conditioning", None)
        uc_blur = kwargs.get("unconditional_guidance_blur", False)
        uc_blur_k = kwargs.get("unconditional_guidance_blur_k", 7)
        uc_blur_rounds = kwargs.get("unconditional_guidance_blur_rounds", int(total_steps/10))
        uc_blur = uc_blur and (t_idx > (total_steps-uc_blur_rounds))
        if uc_blur:
            self.blur = transforms.GaussianBlur(kernel_size=uc_blur_k)  
        
        # Attention Guidance
        attn_guide = kwargs.get("attn_guide", kwargs.get("return_attn", False))
        attn_guide_rounds = kwargs.get("attn_guide_rounds", kwargs.get("return_attn_rounds", 4))
        attn_guide = attn_guide and (t_idx > (total_steps-attn_guide_rounds))
        attn_guide_scale = kwargs.get("attn_guide_scale", 1.1)

        depth_mask = kwargs.get("depth_mask", None)
        if depth_mask is not None:
            x_depth = torch.concat([x[0].cpu(), depth_mask[0].cpu()]).unsqueeze(0)

        e_t_uncond = 0
        e_t_out_attn = 0
        e_factors = []
        e_scales = []
        e_masks = []

        assert "and" in c
        try:        
            pos_factors = c["and"] # "conjunction"
            for (scale, factor, _, mask) in pos_factors:   
                e_factors.append(safe_to(factor, "cuda"))
                e_scales.append(safe_to(scale, "cuda"))
                e_masks.append(safe_to(mask, "cuda"))
            if "not" in c:
                neg_factors = c["not"] # "negation"
                for (scale, factor, _, mask) in neg_factors:
                    e_factors.append(safe_to(factor, "cuda"))
                    e_scales.append(safe_to(-scale, "cuda"))
                    e_masks.append(safe_to(mask, "cuda"))
            # batch inference for each different factor
            bs = 1 + len(e_factors)
            if depth_mask is not None:
                x_in = safe_to(torch.cat([x_depth] * bs), "cuda")
            else:
                x_in = safe_to(torch.cat([x] * bs), "cuda")
            t_in = torch.cat([t] * bs).cuda()
            f_uc = torch.cat([uc.cuda()] + e_factors).cuda()
            if attn_guide:
                out, e_t_out_attn = self._attn_guidance(x_in, t_in, f_uc, **kwargs)                
            else:
                out = self.unet(x_in, t_in, f_uc) 
            if hasattr(out, "sample"):
                out = out.sample           
            e_t_out = list(out.chunk(bs))
            e_t_uncond = e_t_out.pop(0) 
            if uc_blur:
                e_t_uncond = self.blur(e_t_uncond)
            if verbose:         
                self._log_calculate_epsilon(e_t_out[0], e_t_uncond, t)       
        except Exception as e:
            self.log(f"[{self.name}]\t[_calculate_epislon]\t{e}\n{traceback.print_exc()}")
        finally:
            x_in = x_in.cpu()
            t_in = t_in.cpu()
            f_uc = f_uc.cpu()
    
        e_t_sum = sum([safe_to(e_masks[i], dtype="float16", device="cuda") * 
                       safe_to(e_scales[i], dtype="float16", device="cuda") * 
                       (
                        safe_to(e_t, dtype="float16", device="cuda") - 
                        safe_to(e_t_uncond, dtype="float16", device="cuda")
                       ) for i, e_t in enumerate(e_t_out)])

        if attn_guide:
            e_t_sum = e_t_out_attn + attn_guide_scale * (e_t_sum - e_t_out_attn)

        return e_t_uncond, e_t_sum

    @torch.no_grad()
    def _epsilon_t(self, x, c, t, **kwargs):
        verbose = kwargs.get("verbose", False)        
        bar = kwargs.get("bar", None)
        
        # Unconditional Conditioning Scale    
        uc_scale = kwargs.get("unconditional_guidance_scale", 1.)        
        # Time
        t_idx = kwargs.get("t_idx", 0)
        total_steps = kwargs.get("total_steps", 1000)
        # Unconditional Conditioning Scale Decay
        decay_scale = kwargs.get("decaying_uc_scale", False)
        decay_scale_min = kwargs.get("decaying_uc_scale_min", 0)
        decay_scale_start = kwargs.get("decaying_uc_scale_start", total_steps)
        if decay_scale and decay_scale_start < t_idx:
            decay_scale_start = min(t_idx, decay_scale_start)
            uc_scale = max(
                decay_scale_min, 
                uc_scale - (
                    uc_scale * (
                        np.log(t_idx+1-decay_scale_start) /
                        np.log(total_steps)
                    )
                )
            )
        
        # Score corrector
        score_corrector = kwargs.get("score_corrector", None)
        corrector_kwargs = kwargs.get("corrector_kwargs", {})
        corrector_kwargs["verbose"] = verbose
        # Dynamic Scale Clipping post-process
        dynamic_scale_clip = kwargs.get("dynamic_scale_clip", False)
        dynamic_scale_clip_threshold = kwargs.get("dynamic_scale_clip_threshold", 99.9)

        e_t_uncond, e_t_sum = self._calculate_epsilon(x, c, t, **kwargs)

        if dynamic_scale_clip:                                     
            scale_e_t = dynamic_clip(uc_scale * e_t_sum, 
                                     threshold=dynamic_scale_clip_threshold)
        else:
            scale_e_t = uc_scale * e_t_sum
    
        e_t = e_t_uncond + scale_e_t
        if score_corrector is not None:
            e_t = score_corrector.modify_score(e_t, x, t, c, **corrector_kwargs)        

        self.clog("calculate_epsilon", "done")
        return e_t

    @torch.enable_grad()
    def _get_clip_guide(self, x, t, c, e_t_original, **kwargs):
        clip_guidance = kwargs.get("clip_guidance", False)
        clip_guidance_loss_scale = kwargs.get("clip_guidance_loss_scale", 1)
        clip_guidance_grad_scale = kwargs.get("clip_guidance_grad_scale", 1)
        clip_guidance_freq = kwargs.get("clip_guidance_freq", 1)

        if (
            clip_guidance == False 
            or clip_guidance_loss_scale == 0
            or clip_guidance_freq == 0
            or clip_guidance_grad_scale == 0
            or t % clip_guidance_freq != 0
        ):
            return e_t_original, x

        clip_guidance_mode = kwargs.get("clip_guidance_mode", 0)
        clip_guidance_embedding = kwargs.get("clip_guidance_embedding")
        clip_guidance_factor_limit = kwargs.get("clip_guidance_factor_limit", 1)
        clip_guidance_iterative = kwargs.get("clip_guidance_iterative", False)
        txt = kwargs.get("prompt_txt", "")

        set_requires_grad(self.unet, True)  # unet
        set_requires_grad(self.vae, True)  # vae
        x = x.detach().requires_grad_()
        
        depth_mask = kwargs.get("depth_mask", None)
        if depth_mask is not None:
            x_depth = torch.concat([x[0].cpu(), depth_mask[0].cpu()]).unsqueeze(0)

        e_factors = []
        e_scales = []
        e_masks = []                
        try:
            if "and" in c:
                pos_factors = c["and"] # "conjunction"
                for (scale, factor, _, mask) in pos_factors:   
                    if len(e_factors) < clip_guidance_factor_limit:
                        e_factors.append(safe_to(factor, "cuda"))
                        e_scales.append(safe_to(scale, "cuda"))
                        e_masks.append(safe_to(mask, "cuda"))
            if "not" in c:
                neg_factors = c["not"] # "negation"
                for (scale, factor, _, mask) in neg_factors:
                    if len(e_factors) < clip_guidance_factor_limit:
                        e_factors.append(safe_to(factor, "cuda"))
                        e_scales.append(-safe_to(scale, "cuda"))
                        e_masks.append(safe_to(mask, "cuda"))
            # batch inference for each different factor
            # - big vram memory hog 
            #   - limit factors to lower quality/gain vram
            #   - enable iterative to keep quality/lose time
            if clip_guidance_iterative:
                bs = 1
                e_t_out = []
                for factor in e_factors:
                    if depth_mask is not None:
                        x_in = safe_to(torch.cat([x_depth] * bs), "cuda")
                    else:
                        x_in = safe_to(torch.cat([x] * bs), "cuda")
                    t_in = t
                    f_c_in = factor.cuda()
                    with torch.autocast("cuda"):
                        out = self.unet(x_in, t_in, f_c_in)
                    if hasattr(out, "sample"):
                        out = out.sample
                    e_t_out.append(out)
            else:
                bs = len(e_factors)
                if depth_mask is not None:
                    x_in = safe_to(torch.cat([x_depth] * bs), "cuda")
                else:
                    x_in = safe_to(torch.cat([x] * bs), "cuda")
                t_in = torch.cat([t] * bs).cuda()
                f_c_in = torch.cat(e_factors).cuda()
                with torch.autocast("cuda"):
                    out = self.unet(x_in, t_in, f_c_in)
                if hasattr(out, "sample"):
                    out = out.sample
                e_t_out = list(out.chunk(bs))
            e_t = sum([e_masks[i] * e_scales[i] * e_t.half() for i, e_t in enumerate(e_t_out)])                    
        except Exception as e:
            print(f"[_get_clip_guide] t:{t}\n{e}\n{traceback.print_exc()}")
        finally:
            x_in = x_in.cpu()
            t_in = t_in.cpu()
            f_c_in = f_c_in.cpu()            
    
        t_idx = kwargs.get("t_idx", 0)
        ac_t = safe_to(self.scheduler.alphas_cumprod_t[t_idx], device="cuda", dtype=x.dtype)

        if clip_guidance_mode == 0:
            sample = (x - (1 - ac_t).sqrt() * e_t) / ac_t.sqrt()
        else:
            tmp_kwargs = kwargs.copy()
            tmp_kwargs["deterministic"] = True
            tmp_kwargs["clip_sample"] = False
            sample, _ = self.scheduler.step(x, e_t, t_idx, **tmp_kwargs)
        sample = 1 / 0.18215 * sample
        try:
            self.vae.decoder.cuda()
            self.vae.post_quant_conv.cuda()

            img = self._decode(sample)  # vae.decode
            img = (img / 2 + 0.5).clamp(0, 1)
            img = transforms.Resize((224, 224))(img)
            img = self.normalize(img)

            self.clip_model.half().cuda()
            set_requires_grad(self.clip_model, False)
            with torch.autocast("cuda"):
                image_embeddings_clip = self.clip_model.get_image_features(img).float()
                text_embeddings_clip = safe_to(clip_guidance_embedding, device="cuda")      
                mean_embeddings_clip = (image_embeddings_clip + text_embeddings_clip) / 2
                diff_embeddings_clip = image_embeddings_clip - text_embeddings_clip

            image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(
                p=2, dim=-1, keepdim=True
            )      
            text_embeddings_clip = text_embeddings_clip / text_embeddings_clip.norm(
                p=2, dim=-1, keepdim=True
            )
            mean_embeddings_clip = mean_embeddings_clip / mean_embeddings_clip.norm(
                p=2, dim=-1, keepdim=True
            )      
            diff_embeddings_clip = diff_embeddings_clip / diff_embeddings_clip.norm(
                p=2, dim=-1, keepdim=True
            )

            loss = (
                spherical_dist_loss(image_embeddings_clip, text_embeddings_clip).mean()
                * clip_guidance_loss_scale
            )
            loss += (
                spherical_dist_loss(mean_embeddings_clip, text_embeddings_clip).mean()
                * clip_guidance_loss_scale
            )
            loss += (
                spherical_dist_loss(diff_embeddings_clip, text_embeddings_clip).mean()
                * clip_guidance_loss_scale
            )
            loss /= 3
            grads = -torch.autograd.grad(loss.cuda(), x.cuda(), allow_unused=True)[0]
            if clip_guidance_grad_scale > 0:
                grads = grads / grads.norm(p=np.inf, dim=1, keepdim=True)
                grads = grads * (clip_guidance_grad_scale / 100)

            e_t = e_t_original - (1 - ac_t).sqrt() * grads
        except Exception as e:
            self.log(f"[{self.name}]\t[_get_clip_guide]\tloss calculation fail\t{e}")
        finally:
            try:
                image_embeddings_clip = image_embeddings_clip.cpu()
                text_embeddings_clip = text_embeddings_clip.cpu()
                self.clip_model = self.clip_model.cpu().float()
                self.vae.post_quant_conv = self.vae.post_quant_conv.cpu()
                self.vae.decoder = self.vae.decoder.cpu()
                e_t = e_t.detach()
                x = x.detach()
                grads = grads.cpu()
                del grads
                set_requires_grad(self.unet, False)  # unet
                set_requires_grad(self.vae, False)  # vae
            except:
                pass
        return e_t, x

    @torch.no_grad()
    def p_sample_reverse(self, x, c, t, index, **kwargs):
        """Use unet model to predict previous diffusion step.

        Args:
            x (torch.Tensor): latent diffusion field at current step
            c (dict, torch.Tensor, list): prompt conditioning embedding from CLIP 
            t (int): time step
            index (int): index of time step

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor): x_prev, x_pred, epsilon
        """     
        safe_to_x = partial(safe_to, device=x.device, dtype=x.dtype)   
        b, *_, device = *x.shape, x.device

        e_t = self._epsilon_t(x, c, t, **kwargs)

        try:
            e_t, x = self._get_clip_guide(x, t, c, e_t, **kwargs)            
        except Exception as e:
            self.log(f"[{self.name}]\t[CLIP guidance]\t{e}\n{traceback.print_exc()}")
        
        x_prev, pred_x0 = self.scheduler.step(x, e_t, index, **kwargs)
            
        self.clog("p_sample", f"{t} done")
        return x_prev, pred_x0, e_t

    @torch.no_grad()
    def p_sample_forward(self, x, c, t, index, **kwargs):
        """Use unet model to predict next diffusion step

        Args:
            x (torch.Tensor): latent diffusion field at current step
            c (dict, torch.Tensor, list): prompt conditioning embedding from CLIP 
            t (int): time step
            index (int): index of time step

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor): x_prev, x_pred, epsilon
        """           
        
        b, *_, device = *x.shape, x.device

        e_t = self._epsilon_t(x, c, t, **kwargs)
        e_t, x = self._get_clip_guide(x, t, c, e_t, **kwargs)        
                
        x_next, pred_xt = self._get_x_next_and_pred_xt(
            x, e_t, index, **kwargs
        )

        return x_next, pred_xt
    
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.secheduler.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def _get_x_next_and_pred_xt(self, x_next, e_t, t_idx, **kwargs):
        safe_to_x = partial(safe_to, device=x_next.device, dtype=x_next.dtype)
        ac_next_t = safe_to_x(self.scheduler.alphas_cumprod_t[t_idx])
        ac_t = safe_to_x(self.scheduler.alphas_cumprod_prev_t[t_idx])
        xt_weighted = (ac_next_t / ac_t).sqrt() * x_next
        weighted_noise_pred = (
            ac_next_t.sqrt()
            * ((1 / ac_next_t - 1).sqrt() - (1 / ac_t - 1).sqrt())
            * e_t
        )
        x_next = xt_weighted + weighted_noise_pred
        return x_next, weighted_noise_pred

    @torch.no_grad()
    def encode(self, x0, c, t_enc, **kwargs):
        """Deterministic encoding process using unet model.
        `Image -> Noise` that results in the inverse of the normal
        reverse diffusion sampling process.
        Slower than stochastic encoding, but correct for
        reversing this process to obtain original input.

        Args:
            x0 (torch.Tensor): Starting sample, a latent encoded image
            c (list,dict,torch.Tensor): Conditioning, a CLIP embedding of a prompt
            t_enc (int): Encoding time, number of iterations to perform

        Returns:
            tuple(torch.Tensor, dict): x_next, dict containing output of each iteration
        """         
        safe_to_x = partial(safe_to, device=x0.device, dtype=x0.dtype)
        num_reference_steps = self.scheduler.timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        alphas_next = safe_to_x(self.scheduler.alphas_cumprod_t[:num_steps])
        alphas = safe_to_x(self.scheduler.alphas_cumprod_prev_t[:num_steps])

        x_next = x0
        for i in tqdm(range(num_steps), desc="Encoding Image"):
            kwargs["t_idx"] = i
            t = torch.full(
                (x0.shape[0],), i, device=x0.device, dtype=torch.long
            )
            noise_pred = self._epsilon_t(x_next, c, t, **kwargs)
            x_next, xt_pred = self._get_x_next_and_pred_xt(
                x_next, noise_pred, i, **kwargs
            )

        return x_next

    @torch.no_grad()
    def stochastic_encode(self, x0, t, noise=None):
        """Random encoding process using gaussian noise.
        `Image -> Noise` that results in the inverse of the normal
        reverse diffusion sampling process.
        Faster than determninistic `encode`, but can not
        perfectly reproduce input when reverse diffusion is applied
        to this output.

        Args:
            x0 (torch.Tensor): _description_
            t (_type_): _description_
            noise (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """        
        return self.scheduler.add_noise(x0, t, noise=noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, **kwargs):
        """_summary_

        Args:
            x_latent (_type_): _description_
            cond (_type_): _description_
            t_start (_type_): _description_

        Returns:
            _type_: _description_
        """        
        verbose = kwargs.get("verbose", False)
        silent = kwargs.get("silent", False)

        timesteps = self.timesteps[:t_start]

        time_range = np.flip(timesteps)
        
        kwargs["total_steps"] = t_start

        x_dec = x_latent
        if verbose:
            self.log(f"Running {self.name} decode with {t_start} timesteps")

        if silent:
            iterator = iter(time_range)
        else:
            iterator = tqdm(time_range, desc="Decoding image", total=t_start)
            kwargs["bar"] = iterator

        old_eps = []
        old_preds = []
        for i, step in enumerate(iterator):
            index = t_start - i - 1
            ts = torch.full(
                (x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long
            )
            ts_next = torch.full(
                (x_latent.shape[0],),
                time_range[min(i + 1, len(time_range) - 1)],
                device=x_latent.device,
                dtype=torch.long,
            )

            self.clog("decode", f"{i} to p_sample")
            kwargs["old_eps"] = old_eps
            kwargs["t_next"] = ts_next
            x_dec, _, e_t = self.p_sample_reverse(x_dec, cond, ts, index, **kwargs)
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
        self.clog("decode", "done")
        return x_dec

    def sample_img2img(self, steps, x, noise, **kwargs):
        """_summary_

        Args:
            steps (_type_): _description_
            x (_type_): _description_
            noise (_type_): _description_

        Returns:
            _type_: _description_
        """        
        conditioning = kwargs.get("conditioning")
        
        self.timesteps = self.scheduler.set_timesteps(steps, **kwargs)

        denoising_strength = kwargs.get("denoising_strength")
        t_dec = int((1 - denoising_strength) * len(self.timesteps))
        timesteps = self.timesteps[:t_dec]

        encode_steps = kwargs.get("encode_steps", 0)
        if encode_steps == 0:            
            x1 = self.stochastic_encode(x, encode_steps, noise=noise)
            x1 = safe_to(x1, device=x.device, dtype=x.dtype)
        else:
            x1 = self.encode(x, conditioning, encode_steps, **kwargs)

        samples = self.decode(
            x1,
            conditioning,
            t_dec,
            **kwargs
        )

        return samples

    def repaint(self, steps, x, **kwargs):
        original_image = x
        mask_image = kwargs.get("mask")
        num_inference_steps = steps
        eta = kwargs.get("eta", 0.0)
        jump_length = kwargs.get("jump_length", 10)
        jump_n_sample = kwargs.get("jump_n_sample", 10)
        generator = kwargs.get("generator", None)
        
        # sample gaussian noise to begin the loop
        image = torch.randn(
            original_image.shape,
            generator=generator,
            device=self.device,
        )

        # set step values
        self.scheduler.set_timesteps(steps, **kwargs)
        self.scheduler.eta = eta

        t_last = self.scheduler.timesteps[0] + 1
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            if t < t_last:
                # predict the noise residual
                model_output = self._epsilon_t(original_image, c, t, **kwargs)

                model_output, x = self._get_clip_guide(
                    original_image, t, c, model_output, **kwargs
                )
                # compute previous image: x_t -> x_t-1
                image = self.scheduler.step(model_output, t, image, original_image, mask_image, generator).prev_sample

            else:
                # compute the reverse: x_t-1 -> x_t
                image = self.scheduler.undo_step(image, t_last, generator)
            t_last = t

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        return image

def dynamic_clip(x, threshold=99.5):
    s = np.percentile(np.abs(x.cpu()), threshold, axis=tuple(range(1, x.ndim)))
    s = np.max(np.append(s, 1.0))
    return torch.div(x, s)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value
