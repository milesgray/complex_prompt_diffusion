import math
import traceback
import importlib
from typing import Union
from collections import defaultdict
from argparse import Namespace
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torch import autocast

import numpy as np
from einops import rearrange, repeat

from cpd.util import seed_everything, from_json, safe_to
from cpd.metrics import cosine_sim, spherical_dist, euclidean_dist

class ComplexPromptBase:
    def __init__(self, data, 
                 **kwargs):
        """ The following keyword args are used:
        scale: float=1.0, 
        mask=None, 
        model=None, 
        render_options=None, 
        device="cuda",
        logger=print
        """
        if isinstance(data, ComplexPromptBase):
            kwargs = self._copy_other(data, **kwargs) 
            self.parent = data
        else:
            self.log = kwargs.get("logger", print)
            self.parent = None
            while isinstance(data, list):
                data = data[0]
            while hasattr(data, "data"):
                data = data.data
            self.data = data if isinstance(data, str) else str(data)
            self.negative_data = kwargs.get("negative_data", kwargs.get("negative_prompt", ""))
            self.opt = kwargs.get("render_options", {
                "use_fp16": False,
                "H": 512,
                "W": 512,
                "f": 8,
                "C": 4,
                "seed": 1,
                "temperature": 1,
                "scale": 7.5,
                "ddim_eta": 0,
                "n_samples": 1,
            })
            self.scale = kwargs.get("scale", 1)
            self.mask = kwargs.get("mask", 1)
            self.model_dict = kwargs.get("model")
            
            self.vae = self.model_dict["vae"]
            self.unet = self.model_dict["unet"]
            self.embedder = self.model_dict["embedder"]            
            self.tokenizer = self.model_dict["tokenizer"]
            self.embedder_tokenizer = self.model_dict.get("embedder_tokenizer", self.tokenizer)
            self.guide_embedder = self.model_dict["clip_new_model"]
            if "device" not in kwargs and hasattr(self.unet, "time_embed"):
                self.device = self.unet.time_embed[0]._parameters['weight'].device
            else:
                self.device = kwargs.get("device", "cuda")
        use_fp16 = self.opt.use_fp16 if isinstance(self.opt, Namespace) else self.opt["use_fp16"]
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.safe_to = partial(safe_to, device=self.device, dtype=self.dtype)
        self.log(f"Using {self.dtype} as type on {self.device} device")
    
    def __repr__(self):
        return f"{self.__class__.__name__} of '{self.data}' @ {self.scale:0.2f}x with {self.mask.__class__.__name__} style mask"
    
    def _copy_other(self, other, **kwargs):
        self.data = other.data
        self.negative_data = other.negative_data
        self.opt = other.opt
        self.log = other.log
        self.scale = other.scale
        self.mask = other.mask
        self.model_dict = other.model_dict
        self.vae = other.vae
        self.unet = other.unet
        self.embedder = other.embedder
        self.tokenizer = other.tokenizer
        self.guide_embedder = other.guide_embedder
        self.device = other.device
        return kwargs

    def clone(self):
        return ComplexPromptBase(self)

    def to_json(self):
        type_str = str(type(self)).replace("<class '","").replace("'>","")
        module_str = ".".join(type_str.split(".")[:-1])
        class_str = type_str.split(".")[-1]
        return {
            "data": self.data,
            "negative_data": self.negative_data,
            "scale": self.scale,
            #"mask": self.mask, 
            "render_options": self.opt.__dict__ if isinstance(self.opt, Namespace) else self.opt,
            "module": module_str,
            "class": class_str,
            "type": type_str
        }
    
    @classmethod
    def from_json(cls, json: dict, **kwargs):
        model = kwargs.get("model")
        logger = kwargs.get("logger", print)
        data = json.get("data")
        negative_data = json.get("negative_data", "")
        scale = json.get("scale", 1)
        mask = json.get("mask", 1)
        render_options = json.get("render_options", {})
        return cls(data=data, 
                   negative_data=negative_data,
                   scale=scale, 
                   mask=mask, 
                   model=model, 
                   render_options=render_options, 
                   logger=logger)

    def get_embeddings(self, steps: int=1, force: bool=False, verbose: bool=False) -> Tensor:
        if self.built or force:
            return self.embeddings
        else:
            try:                
                self.embeddings = self._build_embeddings(steps=steps, verbose=verbose)
                return self.embeddings
            except Exception as e:
                self.log(f"Failed to build embeddings, returning existing embedding:\t{e}\n{traceback.print_exc()}")
                return self.embeddings

class ComplexPrompt(ComplexPromptBase):
    def __init__(self, data: Union[list,str], 
                 **kwargs): 
        self.__raw_cond_embedding = None
        self.__raw_uncond_embedding = None
        self.__raw_guide_embedding = None    
        super().__init__(data, **kwargs)   
        if isinstance(data, ComplexPrompt):
            kwargs = self._copy_other(data, **kwargs)
        else:         
            self.sampler = kwargs.get("sampler")
            self.tokenizer_config = kwargs.get("tokenizer_config",{
                "truncation": True, 
                "max_length": 77, 
                "return_length": True,
                "return_overflowing_tokens": False, 
                "padding": "max_length", 
                "return_tensors": "pt",
            })
            self._origin_embeddings = self._get_conditioning_embeddings()
            self.embeddings = self._get_conditioning_embeddings()
            self.token_map = self._get_prompt_map()     
            self.path = []
            self.path_embeddings = []
            self.render_buffer = []
        self.built = False
        
    def __repr__(self):
        desc = super().__repr__()        
        desc = f"{desc}\ncontaining:\n"
        desc += '\n'.join([str(p) for p in self.path])
        return desc

    def _copy_other(self, other, **kwargs):
        self.sampler = other.sampler        
        self.tokenizer_config = other.tokenizer_config          
        self._origin_embeddings = tuple(e.clone() for e in other._origin_embeddings)
        self.token_map = other.token_map
        self.path = [p for p in other.path]
        self.path_embeddings = [e for e in other.path_embeddings]
        self.render_buffer = []
        return kwargs

    def clone(self):
        return ComplexPrompt(self)

    def to_json(self):
        json = super().to_json()
        json["path"] = [p.to_json() for p in self.path]        
        
        return json
    
    @classmethod
    def from_json(cls, json: dict, **kwargs):        
        obj = super().from_json(json, **kwargs)
        obj.path = [from_json(p, **kwargs) for p in json.get("path", [])]
        return obj

    def _get_guide_embeddings(self):
        if self.__raw_guide_embedding is None:
            try:
                input_ids = self.tokenizer(self.data).cuda()
            except:
                input_ids = self.tokenizer(self.data,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.cuda()
            self.guide_embedder.cuda()
            with torch.autocast("cuda"):
                self.__raw_guide_embedding = self.guide_embedder.get_text_features(input_ids)
            self.guide_embedder.cpu()
        return self.__raw_guide_embedding

    def _encode(self, data):
        with torch.no_grad():
            if hasattr(self.embedder, "encode"):
                return self.safe_to(self.embedder) \
                        .encode(self.safe_to(data)).cpu()
            else:
                inputs = self.embedder_tokenizer(data,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
                return self.safe_to(self.embedder)(
                    safe_to(input_ids, device=self.device),
                    attention_mask=self.safe_to(attention_mask)
                )[0]

    def _get_conditioning_embeddings(self, guide=False):
        if self.__raw_cond_embedding is None:       
            self.__raw_cond_embedding = self._encode(self.data)
        return (self.__raw_cond_embedding,
                self._get_guide_embeddings())
    
    def _get_unconditional_embeddings(self, batch_size=1):
        if self.__raw_uncond_embedding is None:
            self.__raw_uncond_embedding = self._encode(batch_size * [self.negative_data])  
        return (self.__raw_uncond_embedding, )

    def _get_prompt_map(self):
        try:
            tokenized = self.tokenizer(self.data,  
                                       truncation=True, 
                                       max_length=77, 
                                       return_length=True,
                                       return_overflowing_tokens=False, 
                                       padding="max_length", 
                                       return_tensors="pt")
            token_ids = tokenized["input_ids"].squeeze().cpu()
            return [self.tokenizer.decode(id_) for id_ in token_ids]    
        except:
            token_ids = self.tokenizer(self.data).cpu()
            return token_ids.numpy()

    def get_embeddings(self, steps=1, force=False, verbose=False):
        """
        Get context.

        Tuple of three tensors - CLIP embeddings.
        index 0: primary embedding used as conditioning.
        index 1: secondary CLIP guiding embedding, made with different embedder.
        index 2: unconditional embedding used for negative prompt.
        """
        if force:
            return self.embeddings
        if not self.built:
            try:                
                self.embeddings = self._build_embeddings(steps=steps, verbose=verbose)                
            except Exception as e:
                self.log(f"Failed to build embeddings, returning existing embedding:\t{e}\n{traceback.print_exc()}")
        return self.embeddings

    def _build_embeddings(self, steps=1, verbose=False):
        self.built = False
        self.path_history = []
        self.path_embeddings = []
        self.embeddings = self._get_conditioning_embeddings() + \
                          self._get_unconditional_embeddings()
        self.path_embeddings.append(self.embeddings)
        for p in self.path:
            results = p.apply(self, steps=steps, verbose=verbose)
            for e in results:
                e_main = e[0].cpu().numpy()
                e_guide = e[1].cpu().numpy()
                e_uncon = e[2].cpu().numpy()
                self.path_embeddings.append((e_main, e_guide, e_uncon))
                self.embeddings = (e_main, e_guide, e_uncon)
            
        self.built = True
        return self.embeddings

    def add_transform(self, target, args: dict, transform_cls: callable):
        """
        Add transform object with given target and args to path.

        target (ComplexPrompt): The prompt wrapper that generates the guidance embedding to transform with
        args (dict): contains all parameters for initializing the `transform_cls`
        transform_cls (AbstractTransform): Implements logic to transform between current guidance emebdding and `target`.
        """
        try:
            self.path.append(transform_cls(target=target, 
                                           args=args))
            self.built = False        
        except Exception as e:
            self.log(f'Failed to add transform: {e}\n{traceback.print_exc()}')
        return self

    def add_prompt_lerp(self, prompt: Union[str, callable], args: dict):
        """Add a LerpCLIPEmbeddingTransform to `path` with given prompt target and args."""
        from cpd.embeddings.transforms import LerpCLIPEmbeddingTransform        
        prompt = ComplexPrompt(prompt, 
                               model=self.model_dict,
                               sampler=self.sampler)        
        return self.add_transform(prompt, args, LerpCLIPEmbeddingTransform)

    def add_lerp(self, prompt: Union[str, callable], args: dict):
        """Add a LerpCLIPEmbeddingTransform to `path` with given prompt target and args."""
        return self.add_prompt_lerp(prompt, args)

    def decode(self, z):
        """Convert output of diffusion process to image space."""
        z = 1./0.18215 * z
        if self.opt.use_fp16:
            with autocast("cuda"):
                out = self.vae.decode(z.half())
        else:
            out = self.vae.decode(z)
        if hasattr(out, "sample"):
            out = out.sample
        return out

    def encode(self, x):
        """Convert image space input to latent space."""
        with torch.no_grad():
            dist = self.vae.encode(self.safe_to(x))
        if hasattr(dist, "latent_dist"):
            x = dist.latent_dist.sample()
        else:
            noise = self.safe_to(dist.std) * self.safe_to(torch.randn(dist.mean.shape, device="cpu"))
            x = self.safe_to(dist.mean) + noise
        x = 0.18215 * x
        return x

    def _prepare_sample(self, x: torch.Tensor, coherance: float, diversity: float, reseed: bool=False, renoise: bool=False) -> torch.Tensor:
        if x is None:     
            if reseed: seed_everything(self.opt.seed)   
            z_enc = torch.randn((1,self.opt.z_channels,self.opt.H // 8, self.opt.W // 8), device=self.device)
        else:
            x = sample_from_cv2(x)
            if renoise: x = add_noise(x, 1-coherance)
            if reseed: seed_everything(self.opt.seed, verbose=False)
            x = self.encode(x)
            if renoise: x = sqrt_lerp(x, torch.randn(x.shape, device=x.device), diversity)
            z_enc = x
        return z_enc

    def render(self, 
               lerp_steps=1, 
               sampler=None, 
               steps=None,
               latent=None,
               reset_seed=True,
               verbose=False,
               **kwargs):
        kwargs["verbose"] = verbose
        coherance = kwargs.get("coherance", 0.98)
        diversity = kwargs.get("diversity", 0.00)
        strength = kwargs.get("denoising_strength", 0.65)
        if reset_seed: seed_everything(self.opt.seed)
        with torch.no_grad(), autocast("cuda"):            
            shape = [self.opt.C, self.opt.H // 8, self.opt.W // 8]
            steps = steps if steps else self.opt.steps
            batch_size = self.opt.n_samples
            dtype = "float16" if self.opt.use_fp16 else "float32"
            to_stage = partial(safe_to, device="cpu", dtype=dtype)

            verbose_embed = kwargs.get("verbose_embed", False)
            uc = self._get_unconditional_embeddings(batch_size=batch_size)
            if isinstance(uc, tuple):
                uc = uc[0]
            uc = safe_to(uc, dtype=dtype)
            
            if lerp_steps == 1:
                e = self.get_embeddings(verbose=verbose_embed)
                if isinstance(e, tuple):
                    to_render = [tuple((to_stage(e)
                                    for t in e))]
                elif isinstance(e, dict):
                    e["and"] = [[to_stage(c_e) for c_e in c] 
                                for c in e["and"]]
                    e["not"] = [[to_stage(c_e) for c_e in c]
                                for c in e["not"]]
                    g = e["and"][0][2]
                    u = e["and"][0][3]
                    
                    to_render = [(e,g,u)]
                elif isinstance(e, torch.Tensor):
                    e = to_stage(e)
                    g = safe_to(e, dtype=dtype, clone=True)

                    to_render = [(e,g,u)]
                else:
                    raise ValueError(f"Found bad embedding: {e}")
            else:
                e = self.get_embeddings(steps=lerp_steps, verbose=verbose_embed)
                to_render = safe_to(self.path_embeddings, device="cpu", dtype=dtype)
                # handle composite as last case? or should keep track of ordering...
                if isinstance(e, dict):
                    e["and"] = [[to_stage(c_e) for c_e in c]
                                       for c in e["and"]]
                    e["not"] = [[to_stage(c_e) for c_e in c] 
                                       for c in e["not"]]
                    g = e["and"][0][2]
                    u = e["and"][0][3]
                    
                    to_render.append((e, g, u))

                else:
                    raise ValueError(f"Found bad embedding: {e}")

            sampler = sampler if sampler else self.sampler
            if hasattr(sampler, "sampler"):
                sampler = sampler.sampler
            assert sampler is not None, "Must either pass a sampler into render or assign one when creating prompt"
            
            kwargs["eta"] = self.opt.ddim_eta if "eta" not in kwargs else kwargs["eta"]
            kwargs["temperature"] = self.opt.temperature if "temperature" not in kwargs else kwargs["temperature"]            
            kwargs["unconditional_guidance_scale"] = self.opt.scale if "unconditional_guidance_scale" not in kwargs else kwargs["unconditional_guidance_scale"]
            
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
                                latent = self.prepare_sample(prev_sample, coherance, strength, steps)
                                decode = True
                            kwargs["decode"] = decode                        

                        samples_x = sampler.sample(steps=steps,
                                                   conditioning=c[0],
                                                   clip_guidance_embedding=c[1],
                                                   unconditional_conditioning=uc,
                                                   batch_size=batch_size,
                                                   shape=shape,                                                                                                       
                                                   x_T=latent,
                                                   **kwargs)
                    
                        if isinstance(samples_x, tuple):
                            samples_x = samples_x[0]
                        x = safe_to(self.decode(samples_x), device="cpu", dtype="float32")
                        if lerp_steps > 1:
                            prev_sample = sample_to_cv2(samples_x[1])
                    except Exception as e:
                        print(f"ERROR: {e}\n{traceback.print_exc()}\n")                           
                    finally:
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
                print(f"ERROR: {e}\n{traceback.print_exc()}")
                return (None, samples_x)

class WeightedPrompt(ComplexPrompt):
    def __init__(self, data: Union[list,str], 
                 **kwargs):    
        self.__raw_cond_embedding = None
        self.__raw_guide_embedding = None    
        super().__init__(data, **kwargs)  
        self.subprompts = []
        self.subweights = []

    def __repr__(self):
        desc = super().__repr__()        
        desc = f"{desc}\nwith sub prompts:\n"
        desc += '\n'.join([(w,p) for (w,p) in zip(self.subweights, self.subprompts)])
        return desc

    def _copy_other(self, other):
        super()._copy_other(other)
        if hasattr(other, "subprompts"):
            self.subprompts = other.subprompts
        if hasattr(other, "subweights"):
            self.subweights = other.subweights

    def clone(self):
        return WeightedPrompt(self)

    def to_json(self):
        json = super().to_json()
        if hasattr(self, "subprompts"):
            json["subprompts"] = [p for p in self.subprompts]
        if hasattr(self, "subweights"):
            json["subweights"] = [str(w) for w in self.subweights]
        
        return json
    
    @classmethod
    def from_json(cls, json: dict, **kwargs):        
        obj = super().from_json(json, **kwargs)
        if hasattr(obj, "subprompts"):
            obj.subpompts = [p for p in json.get("subpompts", [])]
        if hasattr(obj, "subweights"):
            obj.subweights = [float(w) for w in json.get("subweights", [])]
        return obj

    def _get_conditioning_embeddings(self, guide=False):
        self.subprompts, self.weights = self._parse_prompt(self.data)
        if len(self.subprompts) == 0:
            return super()._get_conditioning_embeddings(guide=guide)        
        if self.__raw_cond_embedding is None:
            if len(self.subprompts) > 1:
                t = sum(self.weights)
                self.__raw_cond_embedding = sum(
                        [(w/t) * self.embedder.encode(s).cpu() 
                            for w,s in zip(self.weights, self.subprompts)])
            else:
                with torch.no_grad():
                    self.__raw_cond_embedding = self.embedder.encode(self.data).cpu()
        return (self.__raw_cond_embedding,
                self._get_guide_embeddings())

    def _parse_prompt(self, text):
        """
        grabs all text up to the first occurrence of ':' 
        uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
        if ':' has no value defined, defaults to 1.0
        repeats until no text remaining
        """
        remaining = len(text)
        prompts = []
        weights = []
        while remaining > 0:
            if ":" in text:
                idx = text.index(":") # first occurrence from start
                # grab up to index as sub-prompt
                prompt = text[:idx]
                remaining -= idx
                # remove from main text
                text = text[idx+1:]
                # find value for weight 
                if " " in text:
                    idx = text.index(" ") # first occurence
                else: # no space, read to end
                    idx = len(text)
                if idx != 0:
                    try:
                        weight = float(text[:idx])
                    except: # couldn't treat as float
                        print(f"Warning: '{text[:idx]}' is not a value, are you missing a space?")
                        weight = 1.0
                else: # no value found
                    weight = 1.0
                # remove from main text
                remaining -= idx
                text = text[idx+1:]
                # append the sub-prompt and its weight
                prompts.append(prompt)
                weights.append(weight)
            else: # no : found
                if len(text) > 0: # there is still text though
                    # take remainder as weight 1
                    prompts.append(text)
                    weights.append(1.0)
                remaining = 0
        return prompts, weights 
                     
class CompositionalPrompt(ComplexPrompt):     
    def __init__(self, data: Union[list,str], 
                 **kwargs):
        self._conjunctions = []
        self._negations = []
        super().__init__(data, **kwargs)  
            
    def _copy_other(self, other):
        super()._copy_other(other)
        if hasattr(other, "_conjunctions"):
            self._conjunctions = [c for c in other._conjunctions]
        if hasattr(other, "_negations"):
            self._negations = [n for n in other._negations]
        return self

    def clone(self):
        return CompositionalPrompt(self)._copy_other(self)
    
    def to_json(self):
        json = super().to_json()
        json["conjunctions"] = [c.to_json() for c in self._conjunctions]
        json["negations"] = [c.to_json() for c in self._negations]
        return json

    @classmethod
    def from_json(cls, json: dict, **kwargs):        
        obj = super().from_json(json, **kwargs)
        obj._conjunctions = [from_json(p, **kwargs) for p in json.get("conjunctions", [])]
        obj._negations = [from_json(p, **kwargs) for p in json.get("negations", [])]
        return obj

    def _build_embeddings(self, steps=1, verbose=False):
        base_embeddings = super()._build_embeddings(steps=steps, verbose=verbose)

        self.built = False
        try:
            composition = {
                "and": [], 
                "not": []
            }
            composition["and"].append((self.scale, base_embeddings[0], base_embeddings[1], self.mask))
            
            for conj in self._conjunctions:
                embeds = conj.get_embeddings(verbose=verbose)
                composition["and"].append((conj.scale, 
                                            embeds[0], 
                                            embeds[1],
                                            conj.mask))
                
            for neg in self._negations:
                embeds = neg.get_embeddings(verbose=verbose)
                composition["not"].append((neg.scale, 
                                            embeds[0],
                                            embeds[1], 
                                            neg.mask))
                
            self.built = True
            return composition
        except Exception as e:
            if verbose:
                self.log(f"Failed building embeddings:\t{e}\n{traceback.print_exc()}")
            return {
                "and": [(self.scale, base_embeddings[0], base_embeddings[1], self.mask)]
            }
    
    def _update_history_compose(self, p, mode, verbose=False):
        assert mode in ["conjunction", "negation"]
        if verbose: 
            self.log(f"[{p.scale}x]\t{mode.upper()} added: {p.prompt}")
        _edist = euclidean_dist(self.embeddings, p.get_embeddings(), reduce=True)
        _sdist = spherical_dist(self.embeddings, p.get_embeddings(), reduce=True)
        self.path_history.append({"prompt": p.prompt, 
                                "mode": mode,
                                "euler_dist": _edist,
                                "sphere_dist": _sdist,})
        return len(self.path_history)
    
    def add_conjunction(self, prompt: Union[ComplexPrompt,str], 
                        scale: Union[float,None]=1, 
                        mask: Union[torch.Tensor,np.ndarray,None]=1) -> None:
        try:
            if isinstance(prompt, str):
                prompt = ComplexPrompt(prompt, 
                                        scale=scale, 
                                        mask=mask, 
                                        model=self.model_dict,
                                        sampler=self.sampler, 
                                        render_options=self.opt)
            prompt.scale = scale if scale is not None else prompt.scale
            prompt.mask = mask if mask is not None else prompt.mask           
            self._conjunctions.append(prompt)
            self.built = False
        except Exception as e:
            self.log(f'Failed to add conjunction: {e}\n{traceback.print_exc()}')
        return self

    def add_negation(self, prompt: Union[ComplexPrompt,str], 
                     scale: Union[float,None]=1,
                     mask: Union[torch.Tensor,np.ndarray,None]=1) -> None:
        try:
            if isinstance(prompt, str):
                prompt = ComplexPrompt(prompt, 
                                        scale=scale, 
                                        mask=mask, 
                                        model=self.model_dict,
                                        sampler=self.sampler,
                                        render_options=self.opt)
            prompt.scale = scale if scale is not None else prompt.scale
            prompt.mask = mask if mask is not None else prompt.mask
            self._negations.append(prompt)
            self.built = False
        except Exception as e:
            self.log(f'Failed to add negation: {e}\n{traceback.print_exc()}')
        return self

    def add_filter(self, prompt, strength=1.0, mask=1):
        if strength == 0:
            return self
        elif strength > 0:
            return self.add_conjunction(prompt, scale=strength, mask=mask)
        else:
            return self.add_negation(prompt, scale=abs(strength), mask=mask)
    
    def add_masked_filter(self, prompt, mask, strength=1.0):
        """ Add a filter with a mask.
        `mask` can be a string that follows a specific format to programatically describe the mask.
        <direction>_<size>_<minority>
        - direction: which area is being described?
            - left. right, top, bot
            - l, r, t, b
        - size: how much of the <direction> is being described? Small amounts
            - half, third, quarter, fifth, sixth, seventh, eigth, ninth, tenth
            - 2, 3, 4, 5, 6, 7, 8, 9, 10
        - minority: is the <direction> <size> valid or hidden?
            - valid, hidden
            - v, h
        """
        if isinstance(mask, str):
            mask = self._parse_mask_style(mask)
        if len(mask.shape) < 4:
            mask = mask.reshape(1,1,mask.shape[-2], mask.shape[-1])
        return self.add_filter(prompt, strength=strength, mask=mask)
    
    def _parse_mask_style(self, mask_style):
        SIZE_CODES = [
            ("0"),
            ("1"),
            ("2", "half"),
            ("3", "third"),
            ("4", "quarter", "fourth"),
            ("5", "fifrth"),
            ("6", "sixth"),
            ("7", "seventh"),
            ("8", "eigth"),
            ("9", "ninth"),
            ("10", "tenth")
        ]
        DIRECTION_CODES = {
            "top": ("top", "t", "north"),
            "bottom": ("bottom", "bot", "b", "south"),
            "left": ("left", "l", "west"),
            "right": ("right", "r", "east")
        }
        MINORITY_CODES = {
            "hidden": ("hidden", "hide", "h"),
            "valid": ("valid", "visible", "show", "v")
        }
        style_parts = mask_style.split("_")
        style_direction = style_parts[0]
        style_size = style_parts[1] if len(style_parts) > 1 else "half"
        style_minority = style_parts[2] if len(style_parts) > 2 else "valid"
        
        if style_size in SIZE_CODES[2]:
            minor_ratio = 1/2
            major_ratio = 1/2
        elif style_size in SIZE_CODES[3]:
            minor_ratio = 1/3
            major_ratio = 2/3
        elif style_size in SIZE_CODES[4]:
            minor_ratio = 1/4
            major_ratio = 3/4
        elif style_size in SIZE_CODES[5]:
            minor_ratio = 1/5
            major_ratio = 4/5
        elif style_size in SIZE_CODES[6]:
            minor_ratio = 1/6
            major_ratio = 5/6
        elif style_size in SIZE_CODES[7]:
            minor_ratio = 1/7
            major_ratio = 6/7
        elif style_size in SIZE_CODES[8]:
            minor_ratio = 1/8
            major_ratio = 7/8
        elif style_size in SIZE_CODES[9]:
            minor_ratio = 1/9
            major_ratio = 8/9
        elif style_size in SIZE_CODES[10]:
            minor_ratio = 1/10
            major_ratio = 9/10

        if style_minority in MINORITY_CODES["valid"]:
            valid_ratio = minor_ratio
            hidden_ratio = major_ratio
        elif style_minority in MINORITY_CODES["hidden"]:
            valid_ratio = major_ratio 
            hidden_ratio = minor_ratio           

        def floor(dims, ratio):
            return int(math.floor(dims * ratio))
        def ceil(dims, ratio):
            return int(math.ceil(dims * ratio))
        
        shape = (1, self.opt.H // 8, self.opt.W // 8)
        if mask_style == "perspective":
            assert self.opt.H == self.opt.W
            mask = torch.flipud(torch.eye(self.opt.H // 8)) + torch.eye(self.opt.H // 8)
        if style_direction in DIRECTION_CODES["left"]: 
            valid = torch.ones((shape[0], shape[1], floor(shape[2], valid_ratio))).to(torch.uint8)
            hidden = torch.zeros((shape[0], shape[1], ceil(shape[2], hidden_ratio))).to(torch.uint8)
            if valid.shape[2] < hidden.shape[2]:
                mask = torch.concat([valid, hidden], dim=2)
            elif valid.shape[2] > hidden.shape[2]:
                mask = torch.concat([hidden, valid], dim=2)
            else:
                if style_minority in MINORITY_CODES["valid"]:
                    mask = torch.concat([valid, hidden], dim=2)
                else:
                    mask = torch.concat([hidden, valid], dim=2)
        elif style_direction in DIRECTION_CODES["right"]:
            valid = torch.ones((shape[0], shape[1], floor(shape[2], valid_ratio))).to(torch.uint8)
            hidden = torch.zeros((shape[0], shape[1], ceil(shape[2], hidden_ratio))).to(torch.uint8)
            if valid.shape[2] > hidden.shape[2]:
                mask = torch.concat([valid, hidden], dim=2)
            elif valid.shape[2] < hidden.shape[2]:
                mask = torch.concat([hidden, valid], dim=2)
            else:
                if style_minority in MINORITY_CODES["hidden"]:
                    mask = torch.concat([valid, hidden], dim=2)
                else:
                    mask = torch.concat([hidden, valid], dim=2)
        elif style_direction in DIRECTION_CODES["top"]:
            valid = torch.ones((shape[0], floor(shape[1], valid_ratio), shape[2])).to(torch.uint8)
            hidden = torch.zeros((shape[0], ceil(shape[1], hidden_ratio), shape[2])).to(torch.uint8)            
            if valid.shape[1] < hidden.shape[1]:
                mask = torch.concat([valid, hidden], dim=1)
            elif valid.shape[1] > hidden.shape[1]:
                mask = torch.concat([hidden, valid], dim=1)
            else:
                if style_minority in MINORITY_CODES["valid"]:
                    mask = torch.concat([valid, hidden], dim=1)
                else:
                    mask = torch.concat([hidden, valid], dim=1)
        elif style_direction in DIRECTION_CODES["bottom"]:
            valid = torch.ones((shape[0], floor(shape[1], valid_ratio), shape[2])).to(torch.uint8)
            hidden = torch.zeros((shape[0], ceil(shape[1], hidden_ratio), shape[2])).to(torch.uint8)            
            if valid.shape[1] > hidden.shape[1]:
                mask = torch.concat([valid, hidden], dim=1)
            elif valid.shape[1] < hidden.shape[1]:
                mask = torch.concat([hidden, valid], dim=1)
            else:
                if style_minority in MINORITY_CODES["hidden"]:
                    mask = torch.concat([valid, hidden], dim=1)
                else:
                    mask = torch.concat([hidden, valid], dim=1)
        assert mask.shape == shape, f"auto created mask is invalid shape {mask.shape}, should be {shape}"
        return mask 


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

def sqrt_lerp(x, y,  a):
    return (1-a)*x+np.sqrt(a)*y