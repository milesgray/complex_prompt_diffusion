import traceback
from typing import Union
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torch import autocast
import numpy as np
from einops import rearrange

from cpd.util import seed_everything, from_json, safe_to

class RenderEngine:
    def __init__(self, model_mgr, sampler, args):
        self.points = OrderedDict()
        self.location = 0
        self.sampler = sampler
        self.args = args
        self.model_mgr = model_mgr
        self.dtype = torch.float16 if args.use_fp16 else torch.float32
        self.device = args.device
        self.safe_to = partial(safe_to, device=args.device, dtype=args.dtype)

    def decode(self, z):
        z = 1./0.18215 * z
        return self.model_mgr.decode(z)

    def encode(self, x):
        with torch.no_grad():
            dist = self.vae.encode(x.cuda())
        noise = self.safe_to(dist) * self.safe_to(torch.randn(dist.mean.shape, device="cpu"))
        x = self.safe_to(dist.mean) + noise
        x = 0.18215 * x
        return x
        
    def _build_render_queue(self, lerp_steps, **kwargs):
        verbose_embed = kwargs.get("verbose_embed", False)
        batch_size = kwargs.get("batch_size", 1)
        to_stage = partial(safe_to, device="cpu", dtype=self.dtype)
        uc = self.model_mgr.get_unconditional_embeddings(batch_size)
    
        last_emb = self.points[self.location].get_embeddings(steps=lerp_steps, 
                                        verbose=verbose_embed)
        to_render = [to_stage(e)
                        for e in self.points[self.location].path_embeddings]
        # handle composite as last case? or should keep track of ordering...
        if isinstance(last_emb, dict):
            last_emb["and"] = [(to_stage(c[0]), 
                                to_stage(c[1]), 
                                to_stage(c[3]))
                            for c in last_emb["and"]]
            last_emb["not"] = [(to_stage(c[0]), 
                                to_stage(c[1]), 
                                to_stage(c[3]))
                            for c in last_emb["not"]]
            guide_emb = to_stage(last_emb["and"][0][2])
            
            to_render.append((last_emb, guide_emb))

        uc = to_stage(uc)

        return uc, to_render

    def _prepare_sample(self, x: torch.Tensor, coherance: float, diversity: float, reseed: bool=False, renoise: bool=False) -> torch.Tensor:
        if x is None:     
            if reseed: seed_everything(self.args.seed)   
            z_enc = torch.randn((1,self.args.z_channels,self.args.H // 8, self.args.W // 8), device=self.device)
        else:
            if reseed: seed_everything(self.args.seed, verbose=False)
            x = sample_from_cv2(x)
            if renoise: x = add_noise(x, 1-coherance)
            if reseed: seed_everything(self.args.seed, verbose=False)
            x = self.encode(x)
            if reseed: seed_everything(self.args.seed, verbose=False)
            if renoise: x = sqrt_lerp(x, torch.randn(x.shape, device=x.device), diversity)
            z_enc = x
        return z_enc

    def render(self, 
               lerp_steps=1, 
               sampler=None, 
               steps=None,
               start_code=None,
               reset_seed=True,
               verbose=False,
               **kwargs):
        coherance = kwargs.get("coherance", 0.98)
        diversity = kwargs.get("diversity", 0.00)
        strength = kwargs.get("denoising_strength", 0.65)
        if reset_seed: seed_everything(self.args.seed)
        with torch.no_grad(), autocast("cuda"):            
            shape = [self.args.C, self.args.H // 8, self.args.W // 8]
            steps = steps if steps else self.args.steps
            batch_size = self.args.n_samples

            uc, to_render = self._build_render_queue(lerp_steps, **kwargs)

            sampler = sampler if sampler else self.sampler
            if hasattr(sampler, "sampler"):
                sampler = sampler.sampler
            assert sampler is not None, "Must either pass a sampler into render or assign one when creating prompt"
            
            try:
                for i, c in enumerate(to_render):
                    samples_x = None
                    try:
                        if lerp_steps > 1:
                            decode = False
                            print(f"[{i}/{lerp_steps}] RENDERING FRAME {i+1} of {len(to_render)}")
                            if prev_sample is None:
                                latent = torch.randn([batch_size] + shape, device=self.device)
                            else:
                                latent = self._prepare_sample(prev_sample, coherance, diversity)
                                decode = True
                            kwargs["decode"] = decode
                        samples_x = sampler.sample(steps=steps,
                                                    conditioning=c[0],
                                                    clip_guidance_embedding=c[1],
                                                    batch_size=batch_size,
                                                    shape=shape,
                                                    verbose=verbose,
                                                    unconditional_guidance_scale=self.args.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=self.args.ddim_eta,
                                                    temperature=self.args.temperature,
                                                    x_T=latent,
                                                    **kwargs)
                    
                        if isinstance(samples_x, tuple):
                            samples_x = samples_x[0]
                        if lerp_steps > 1:
                            prev_sample = sample_to_cv2(samples_x)
                        x = safe_to(self._decode(samples_x), device="cpu", dtype="float32")
                    except Exception as e:
                        print(f"ERROR: {e}\n{traceback.print_exc()}\n")                           
                    finally:
                        if samples_x is not None:
                            samples_x = safe_to(samples_x, device="cpu")
                        uc = safe_to(uc, device="cpu")
                        c = safe_to(c, device="cpu")
                        
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()       
                    
                    imgs = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                    imgs = imgs.permute(0, 2, 3, 1) \
                            .mul(255) \
                                .to(torch.uint8)

                    for img, sample in zip(imgs.chunk(imgs.shape[0]), x.chunk(x.shape[0])):
                        img = img.squeeze() \
                                .numpy()
                        self.render_buffer.append((img, sample))
                    
                return (img, sample)
            except Exception as e:
                print(f"ERROR: {e}\n{traceback.print_exc()}\n")
                return (None, samples_x)

def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample

def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255)
    return sample_int8.astype(type)      

def add_noise(x: torch.Tensor, strength: float) -> torch.Tensor:
        return x + torch.randn(x.shape, device=x.device) * strength     

def sqrt_lerp(x: torch.Tensor, y: torch.Tensor, a: float) -> torch.Tensor:
    return (1-a)*x+np.sqrt(a)*y                