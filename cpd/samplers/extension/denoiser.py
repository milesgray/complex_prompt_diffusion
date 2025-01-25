
import math
import traceback
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

import numpy as np
from scipy import integrate
from tqdm import trange

from cpd.util import CudaMon, safe_to
from cpd.samplers.registry import register
from cpd.samplers.extension.callbacks import render_callback
from cpd.samplers.extension import create as create_extension
from cpd.scheduler.discrete import DiscreteScheduler, SigmaScheduler

from IPython import display
from PIL import Image

class Denoiser(nn.Module):
    def __init__(self, unet, vae, tokenizer, clip_model, decode, quantize=False, **kwargs):
        super().__init__()
        self.name = kwargs.get("name", "Denoiser")
        self.dtype = next(unet.parameters()).dtype
        self.device = next(unet.parameters()).device
        self.safe_to = partial(safe_to, dtype=self.dtype, device=self.device)

        self.scheduler = SigmaScheduler(
            num_train_timesteps=1000, **kwargs
        )
        self.quantize = quantize
        self.sigma_data = 1.
        
        self.decode = decode        
        self.unet = unet
        self.vae = vae                
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        set_requires_grad(self.clip_model, False)

        # mean: [0.48145466, 0.4578275, 0.40821073]
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        # std: [0.26862954, 0.26130258, 0.27577711]
        image_std = [0.26862954, 0.26130258, 0.27577711]
        self.normalize = transforms.Normalize(mean=image_mean, std=image_std)
        self.blur = transforms.GaussianBlur(kernel_size=31)

        self.collect_tensors = kwargs.get("collect_tensors", False)
        self.feat_list = []
        self.attn_list = []        
        def cap(m, i, o):
            self.feat_list.append(i)
            self.attn_list.append(o)
        def traverse(model, parent):
            for name, layer in model._modules.items():
                full_name = f"{parent}_{name}"
                if "attn" in name and "output" in parent:                    
                    layer.register_forward_hook(cap)
                else:
                    if len(layer._modules) > 0:
                        traverse(layer, full_name)
        if self.collect_tensors:
            traverse(self.unet, "base")
        self.clog = CudaMon("Denoiser")
        self.clog("__init__", "done init")

    def _apply_unet(self, x, sigma, cond,  **kwargs):
        _, c_in = self.scheduler.get_scalings(sigma)
        eps = self.unet(x * c_in, self.scheduler.sigma_to_t(sigma), cond, **kwargs)
        return eps

    @torch.enable_grad()
    def _get_clip_guide(self, x, e_t_original, **kwargs):
        t_idx = kwargs.get("t_idx", 0)
        t = kwargs.get("t", None)        
        c = kwargs.get("conditioning", kwargs.get("c", None))
        sigma = kwargs.get("sigma", None)
        
        clip_guidance = kwargs.get("clip_guidance", False)
        clip_guidance_mode = kwargs.get("clip_guidance_mode", 0)
        clip_guidance_scale = kwargs.get("clip_guidance_scale", 0)        
        clip_guidance_grad_scale = kwargs.get("clip_guidance_grad_scale", 0)
        clip_guidance_factor_limit = kwargs.get("clip_guidance_factor_limit", 1)
        clip_guidance_factor_list = kwargs.get("clip_guidance_factor_list", [])
        clip_guidance_iterative = kwargs.get("clip_guidance_iterative", False)
        clip_guidance_freq = kwargs.get("clip_guidance_freq", 1)

        if t is None:
            assert sigma is not None, "[denoiser]\t[get_clip_guide]\t `sigma` kwarg required when no `t` kwarg supplied"
            t = int(self.scheduler.sigma_to_t(sigma))

        if (
            clip_guidance == False 
                or clip_guidance_scale == 0
                    or clip_guidance_freq == 0
                        or clip_guidance_grad_scale == 0
                            or t % clip_guidance_freq != 0
        ):
            return e_t_original, x

        clip_guidance_embedding = kwargs.get("clip_guidance_embedding")

        set_requires_grad(self.unet, True) # unet
        set_requires_grad(self.vae, True) # vae  
        
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
                for i, (scale, factor, _, mask) in enumerate(pos_factors): 
                    if len(e_factors) < clip_guidance_factor_limit or i in clip_guidance_factor_list:
                        e_factors.append(safe_to(factor, "cuda"))
                        e_scales.append(safe_to(scale, "cuda"))
                        e_masks.append(safe_to(mask, "cuda"))
            if "not" in c:
                neg_factors = c["not"] # "negation"
                for i, (scale, factor, _, mask) in enumerate(neg_factors):
                    if len(e_factors) < clip_guidance_factor_limit or i in clip_guidance_factor_list:
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
                        x_in = [x_depth] * bs
                    else:
                        x_in = [x] * bs
                    _, sc_in = self.scheduler.get_scalings(t)
                    x_in = safe_to(x * sc_in.half(), "cuda")

                    t_in = safe_to(t, "cuda")
                    f_c_in = safe_to(factor, "cuda")                     
                    
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
                _, sc_in = self.scheduler.get_scalings(t)
                x_in = safe_to(x * sc_in.half(), "cuda")

                t_in = safe_to(torch.cat([t] * bs), "cuda")
                f_c_in = safe_to(torch.cat(e_factors), "cuda")  
                
                with torch.autocast("cuda"):              
                    out = self.unet(x_in, t_in, f_c_in)
                if hasattr(out, "sample"):
                    out = out.sample
                e_t_out = list(out.chunk(bs))
            e_t = sum([e_masks[i] * e_scales[i] * e_t.half() for i, e_t in enumerate(e_t_out)])                    
        except Exception as e:
            print(f"[_get_clip_guide] t:{t}\n{e}\n{traceback.print_exc()}")
        finally:
            try:
                out = out.cpu()
                del out
                e_scales = e_scales.cpu()
                del e_scales
                e_masks = e_masks.cpu()
                del e_masks
                x_in = x_in.cpu()
                del x_in
                t_in = t_in.cpu()
                f_c_in = f_c_in.cpu()
                del f_c_in
            except:
                pass

        #sample = x - sigma * e_t  #  sigma vs sc_out?
        tmp_kwargs = kwargs.copy()
        tmp_kwargs["deterministic"] = True
        tmp_kwargs["clip_sample"] = False
        sample, _ = self.scheduler.step(x, e_t, t_idx, **tmp_kwargs)

        sample = 1 / 0.18215 * sample
        try:            
            # use this pointer to the decode method to avoid triggering the 'send to gpu' wrapper lambda
            img = self.vae.decode(sample)
            if hasattr(img, "sample"):
                img = img.sample   #vae.decode
            img = (img / 2 + 0.5).clamp(0, 1)
            img = transforms.Resize((224,224))(img)
            img = self.normalize(img)            

            self.clip_model = self.clip_model.half().cuda()
            set_requires_grad(self.clip_model, False)
            with torch.autocast("cuda"):
                image_embeddings_clip = self.clip_model.get_image_features(img.cuda()).float()
                text_embeddings_clip = safe_to(clip_guidance_embedding, device="cuda")      
                if clip_guidance_mode == 1:
                    mean_embeddings_clip = (image_embeddings_clip + text_embeddings_clip) / 2
                    diff_embeddings_clip = image_embeddings_clip - text_embeddings_clip
        
            image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(
                p=2, dim=-1, keepdim=True
            )      
            text_embeddings_clip = text_embeddings_clip / text_embeddings_clip.norm(
                p=2, dim=-1, keepdim=True
            )
            if clip_guidance_mode == 1:
                mean_embeddings_clip = mean_embeddings_clip / mean_embeddings_clip.norm(
                    p=2, dim=-1, keepdim=True
                )      
                diff_embeddings_clip = diff_embeddings_clip / diff_embeddings_clip.norm(
                    p=2, dim=-1, keepdim=True
                )

            loss = spherical_dist_loss(image_embeddings_clip, text_embeddings_clip).mean()
            if clip_guidance_mode == 1:
                loss += spherical_dist_loss(mean_embeddings_clip, text_embeddings_clip).mean()
                loss += spherical_dist_loss(diff_embeddings_clip, text_embeddings_clip).mean()
                loss /= 3
            loss *= clip_guidance_scale
            
            grads = -torch.autograd.grad(loss.cuda(), x.cuda(), allow_unused=True)[0]
            if clip_guidance_grad_scale > 0:
                grads = grads / grads.norm(p=np.inf, dim=1, keepdim=True)
                grads = grads * (clip_guidance_grad_scale / 100)
            
            x = x.detach() + grads * (sigma**2)
        except Exception as e:
            print(f"[_get_clip_guide] grads\n{e}\n{traceback.print_exc()}")
        finally:
            try:
                sample = sample.cpu()
                img = img.cpu()
                del sample, img
                image_embeddings_clip = image_embeddings_clip.cpu()
                text_embeddings_clip = text_embeddings_clip.cpu()
                del image_embeddings_clip, text_embeddings_clip
                set_requires_grad(self.unet, False) # unet
                set_requires_grad(self.vae, False) # vae
                grads = grads.cpu()
                del grads
                x = x.detach()
                e_t = e_t_original
            except Exception as e:
                print(f"[_get_clip_guide]\t[finally]\n{e}\n{traceback.print_exc()}")
            

        return e_t, x

    def _attn_guidance(self, x_in, t_in, f_uc, **kwargs):
        t_idx = kwargs.get("t_idx")
        x = x_in[0]
        attn_guide_mode = kwargs.get("attn_guide_mode", 2)
        attn_guide_idx = kwargs.get("attn_guide_idx",kwargs.get("return_attn_idx", -1))        
        
        attn_guide_mask_idx = kwargs.get("attn_guide_mask_idx", 0)
        attn_guide_mask_threshold = kwargs.get("attn_guide_mask_threshold", kwargs.get("attn_mask_threshold", 90))
        attn_guide_blur_k = kwargs.get("attn_guide_blur_k", 31)

        attn_guide_blur = transforms.GaussianBlur(kernel_size=attn_guide_blur_k)
                
        scaling_in = kwargs.get("sc_in")
        if attn_guide_mode == 2:
            scaling_in = 1
        out = torch.zeros_like(t_in)
        attn_out = [0]
        t = kwargs.get("t")
        sigma = kwargs.get("sigma")
        uc = self.safe_to(f_uc[0])

        try:
            t_in = self.safe_to(self.scheduler.sigma_to_t(t_in))
            out, attn = self.unet(x_in, t_in, f_uc, return_attn=True)
            if hasattr(out, "sample"):                        
                out = out.sample
            attn = attn[attn_guide_idx]
            if isinstance(attn, tuple):
                attn = attn[0]
            mask = attn.mean(1, keepdims=True)
            s = np.percentile(mask.detach().cpu(), attn_guide_mask_threshold, axis=tuple(range(0,mask.ndim)))
            mask[mask > s] = 1
            mask[mask < s] = 0

            sample = x - (sigma * out[0])
            
            blur_sample = attn_guide_blur(sample)
            
            blur_x = blur_sample + (sigma * out[0])
            
            masked_x = blur_x * mask[0]
            guide_x = masked_x + (x * (1 - mask[0]))
            guide_x = self.safe_to(guide_x * scaling_in)
            attn_out = self.unet(guide_x, t, uc)
            if hasattr(attn_out, "sample"):
                attn_out= attn_out.sample
        except Exception as e:
            self.log(f"[{self.name}]\t[_attn_guidance]\t{e}\n{traceback.print_exc()}")
        finally:
            try:
                t.cpu()
                uc.cpu()
                guide_x = guide_x.cpu()          
            except:
                pass
        return out, attn_out[0]

    def _process_conditioning(self, x, c, sigma, **kwargs):
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
        attn_guide_mode = kwargs.get("attn_guide_mode", 2)
        attn_guide_rounds = kwargs.get("attn_guide_rounds", kwargs.get("return_attn_rounds", 4))
        attn_guide = attn_guide and (t_idx > (total_steps-attn_guide_rounds))
        attn_guide_idx = kwargs.get("attn_guide_idx", kwargs.get("return_attn_idx", -1))
        attn_guide_scale = kwargs.get("attn_guide_scale", 1.1)
        attn_guide_mask_threshold = kwargs.get("attn_guide_mask_threshold", kwargs.get("attn_mask_threshold", 90)  )        
        attn_guide_blur_k = kwargs.get("attn_guide_blur_k", 31)

        attn_guide_blur = transforms.GaussianBlur(kernel_size=attn_guide_blur_k)

        # inject feats/attns
        inject_feats = kwargs.get("inject_feats", None)
        inject_feats_stop = kwargs.get("inject_feats_stop", 10)
        inject_attns = kwargs.get("inject_attns", None)
        inject_attns_stop = kwargs.get("inject_attns_stop", 10)

        depth_mask = kwargs.get("depth_mask", None)
        if depth_mask is not None:
            x_depth = torch.cat([x[0].cpu(), depth_mask[0].cpu()]).unsqueeze(0)
        
        t = self.scheduler.sigma_to_t(sigma).cuda()
        e_t_uncond = None
        e_factors = []
        e_scales = []
        e_masks = []
    
        assert "and" in c
        try:
            if "and" in c:
                pos_factors = c["and"] # "conjunction"
                for (scale, factor, _, mask) in pos_factors:   
                    e_factors.append(self.safe_to(factor))
                    e_scales.append(self.safe_to(scale))
                    e_masks.append(self.safe_to(mask))
            if "not" in c:
                neg_factors = c["not"] # "negation"
                for (scale, factor, _, mask) in neg_factors:
                    e_factors.append(self.safe_to(factor))
                    e_scales.append(self.safe_to(-scale))
                    e_masks.append(self.safe_to(mask))
            # batch inference for each different factor
            bs = 1 + len(e_factors)
            t_in = torch.cat([sigma] * bs).cuda()
            f_uc = torch.cat([uc.cuda()] + e_factors).cuda()
            if depth_mask is not None:                
                _, sc_in = [append_dims(c, x_depth.ndim) for c in self.scheduler.get_scalings(t_in)]
                x_in = safe_to(x_depth, "cuda") * sc_in
            else:
                _, sc_in = [append_dims(c, x.ndim) for c in self.scheduler.get_scalings(t_in)]
                x_in = safe_to(x, "cuda") * sc_in
            x_in = x_in.cuda()
            t_in = self.safe_to(self.scheduler.sigma_to_t(t_in))
            # attn returned to make saliency mask
            # inject_feats - list of feats to use in upsample half of unet
            # inject_attns - list of attn tensors to use in upsample half of unet
            out, attn = self.unet(x_in, t_in, f_uc, 
                                  return_attn=True, 
                                  inject_feats=inject_feats, 
                                  inject_feats_stop=inject_feats_stop,
                                  inject_attns=inject_attns,
                                  inject_attns_stop=inject_attns_stop)
            if hasattr(out, "sample"):                        
                out = out.sample
            attn = attn[attn_guide_idx]
            if isinstance(attn, tuple):
                attn = attn[0]
            mask = attn.mean(1, keepdims=True)
            
            if attn_guide:
                s = np.percentile(mask.detach().cpu(), attn_guide_mask_threshold, axis=tuple(range(0,mask.ndim)))
                mask[mask > s] = 1
                mask[mask < s] = 0
                mask = mask[0]
                gamma = kwargs.get("gamma", 0)
                sigma_hat = sigma * (gamma + 1)
                if gamma > 0:
                    x = x + out[0] * (sigma_hat**2 - sigma*2)**0.5
                sample = x - (sigma_hat * out[0])
                blur_sample = attn_guide_blur(sample)
                blur_x = blur_sample + (sigma_hat / out[0])            
                if gamma > 0:
                    x = x - out[0] / (sigma_hat**2 - sigma*2)**0.5                
                masked_x = blur_x * mask
                if attn_guide_mode == 2:
                    masked_x = (masked_x * sc_in[0]).cuda()
                guide_x = masked_x + (x * (1 - mask))            
                if attn_guide_mode == 1:
                    guide_x = (guide_x * sc_in[0]).cuda()
                attn_out = self.unet(guide_x, t, uc.cuda())
                if hasattr(attn_out, "sample"):
                    attn_out= attn_out.sample
                guide_x = guide_x.cpu()
                uc = uc.cpu()
                e_t_out_attn = attn_out[0]
            
            if hasattr(out, "sample"):
                out = out.sample
            e_t_out = list(out.chunk(bs))
            e_t_uncond = e_t_out.pop(0) 
            if uc_blur:
                e_t_uncond = self.blur(e_t_uncond)               
        except Exception as e:
            print(f"[{self.name}]\t[_processing_conditioning]\t{e}\n{traceback.print_exc()}")
        finally:
            x_in = x_in.cpu()
            t_in = t_in.cpu()
            f_uc = f_uc.cpu()
        
        sum_e_t = sum([            
            safe_to(e_masks[i], dtype="float16", device="cuda") * 
            safe_to(e_scales[i], dtype="float16", device="cuda") * 
            (
                (   
                    safe_to(e_t, dtype="float16", device="cuda")
                ) - 
                (           
                    safe_to(e_t_uncond, dtype="float16", device="cuda")
                )
            ) for i, e_t in enumerate(e_t_out)])
        if attn_guide:
            sum_e_t = e_t_out_attn + attn_guide_scale * (sum_e_t - e_t_out_attn)
        return sum_e_t, e_t_uncond

    @torch.no_grad()
    def _calculate_epsilon(self, x, **kwargs):
        self.feat_list = []
        self.attn_list = []
        sigma = kwargs.get("sigma", None)
        c = kwargs.get("conditioning", None)
        if isinstance(c, list):
            c = c[t_idx]
        # Unconditional Conditioning Scale
        uc = kwargs.get("unconditional_conditioning", None)        
        uc_scale = kwargs.get("unconditional_guidance_scale", 1.) 
        
        # Time
        t_idx = kwargs.get("t_idx", 0)
        total_steps = kwargs.get("total_steps", 1000)
        # Unconditional Conditioning Scale Decay
        decay_scale = kwargs.get("decaying_uc_scale", False)
        decay_scale_min = kwargs.get("decaying_uc_scale_min", 2)
        decay_scale_start = kwargs.get("decaying_uc_scale_start", int(total_steps * 0.2))
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
        # Score Corrector
        score_corrector = kwargs.get("score_corrector", None)
        corrector_kwargs = kwargs.get("corrector_kwargs", {})
        # Dynamic Scale Clip
        scaled_clip = kwargs.get("scaled_clip",kwargs.get("dynamic_scale_clip", False))
        scaled_clip_alg = kwargs.get("scaled_clip_alg", "dynamic_thresholding")
        scaled_clip_threshold = kwargs.get("scaled_clip_threshold", kwargs.get("dynamic_scale_clip_threshold", 99.5))
            
        # Verbose/logging
        verbose = kwargs.get("verbose", False)
        corrector_kwargs["verbose"] = verbose
        bar = kwargs.get("bar", None)
        
        sum_e_t, e_t_uncond = self._process_conditioning(x, c, sigma, **kwargs)

        if scaled_clip:
            scaled_thresholding = create_extension(scaled_clip_alg)        
            scaled_e_t = scaled_thresholding(uc_scale * sum_e_t, threshold=scaled_clip_threshold)
        else:
            scaled_e_t = uc_scale * sum_e_t
        e_t = e_t_uncond + scaled_e_t
            
        if score_corrector is not None:
            e_t = score_corrector.modify_score(e_t, x, self.scheduler.sigma_to_t(sigma), c, **corrector_kwargs)        

        self.clog("calculate_epsilon", "done")
        return e_t

    def complex_epsilon(self, x, **kwargs):        
        eps = self._calculate_epsilon(x.clone(), **kwargs)        
        eps, x = self._get_clip_guide(x, eps, **kwargs)
        return eps, x
        
    def forward(self, x, sigma, **kwargs):    
        assert x.shape[1] == 4, f"[denoiser]\t[forward]\t `x` has incorrect number of channels: {x.shape}"    
        kwargs["sigma"] = sigma
        eps, x = self.complex_epsilon(x, **kwargs)   
        assert eps.shape == x.shape, f"[denoiser]\t[forward]\t Shape mismatch. x: {x.shape} eps: {eps.shape}"                                                                                                                          
        gamma = kwargs.get("gamma", 0)
        sigma_hat = sigma * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigma*2)**0.5
        pred_type = kwargs.get("pred_type", "epsilon")
        if pred_type == "epsilon":
            
            sample = x - sigma_hat * eps 
        elif pred_type == "velocity":
            sample = eps * (-sigma / (sigma**2 + 1) ** 0.5) + (x / (sigma**2 + 1))
        render_callback(sample, sigma, **kwargs)
        return sample


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def append_dims(x, target_dims):
    """Append dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def dynamic_clip(x, threshold=99.5):
    s = np.percentile(np.abs(x.cpu()), threshold, axis=tuple(range(1,x.ndim)))
    s = np.max(np.append(s,1.0))
    return torch.div(x, s)

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value 