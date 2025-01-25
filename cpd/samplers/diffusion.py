from functools import partial

import torch

from cpd.util import safe_to

class DiffusionSampler:
    def __init__(self, model, **kwargs):
        self.dtype = next(model["unet"].parameters()).dtype
        self.device = "cuda" #next(model["unet"].parameters()).device
        self.safe_to = partial(safe_to, dtype=self.dtype, device=self.device)

        # core diffusion model - estimates noise transitions conditioned on a timestep
        self.unet = model["unet"]
        # encodes the input to the unet and decodes the final output, upsamples 8x
        self.vae = model["vae"]
        # CLIP tokenizer that turns text into a series of ints that the CLIP encoder expects
        self.tokenizer = model["tokenizer"]

        # Reference to the VAE's decode method to avoid hot-swapping to/from GPU
        # optimization that destroys gradients. Used for CLIP conditioning.
        self._decode = model["decode"]
        # Improved CLIP model that outputs 512 size embeddings instead of 768
        # (as the unet expects 768, it can only be used to generate gradients,
        # not directly used as the conditioning input to the unet)
        self.clip_model = model["clip_new_model"]
        set_requires_grad(self.clip_model, False)

    @torch.no_grad()
    def sample(self, steps, batch_size, shape, **kwargs):
        return torch.randn(shape)
    
    @torch.no_grad()
    def encode(self, x0, c, t_enc, **kwargs):
        x_next = x0 + torch.randn(x0.shape)
        inter_steps = []
        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        return x_next, out
    
    @torch.no_grad()
    def stochastic_encode(self, x0, t, noise=None):
        return x0

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, **kwargs):
        return x_latent

    def sample_img2img(self, steps, x, noise, **kwargs):
        return x

class DiffusionSamplerWrapper:
    def __init__(self, name: str, **kwargs):
        constructor = kwargs.get("constructor", DiffusionSampler)
        self.sampler = constructor(kwargs.get("model"))
        self.name = name
        self.batch_size = kwargs.get("batch_size", 1)
        self.width = kwargs.get("width", 512)
        self.height = kwargs.get("height", 512)
        self.z_channels = kwargs.get("z_channels", 4)
        self.scale = kwargs.get("scale", 7.5)
        self.use_start_code = kwargs.get("use_start_code", False)
        self.steps = kwargs.get("steps", 50)
        self.eta = kwargs.get("eta", 0)
        self.temperature = kwargs.get("temperature", 1)
        self.denoising_strength = kwargs.get("denoising_strength", 0.0)

    def to_json(self):
        return {
            "name": self.name,
            "args": {
                "batch_size": self.batch_size,
                "width": self.width,
                "height": self.height,
                "z_channels": self.z_channels,
                "scale": self.scale,
                "use_start_code": self.use_start_code,
                "steps": self.steps,
                "eta": self.eta,
                "temperature": self.temperature,
                "denoising_strength": self.denoising_strength,
            }
        }

    def sample(self, 
               conditioning: torch.Tensor=None, 
               **kwargs):
        
        verbose = kwargs.get("verbose",False)
        shape = [self.z_channels, self.width // 8, self.height // 8]
        if self.use_start_code:
            if start_code is None:
                start_code = torch.randn((self.batch_size,) + shape)
        else:
            start_code = None

        kwargs["unconditional_guidance_scale"]=self.scale
        kwargs["eta"] = self.eta
        kwargs["temperature"] = self.temperature
        kwargs["x_T"] = start_code
    
        result = self.sampler.sample(steps=self.steps,
                                        conditioning=conditioning,
                                        batch_size=self.batch_size,
                                        shape=shape,
                                        **kwargs)
                                    
        if isinstance(result, tuple):
            samples = result[0]
        else:
            samples = result
        return samples

    def sample_img(self, img, mask, 
                   conditioning=None,    
                   unconditional_conditioning=None, 
                   noise=None):
        self.sampler.make_schedule(num_steps=self.steps, eta=self.eta, verbose=False)
    
        t_enc = int(min(self.denoising_strength, 0.999) * self.steps)
        t = torch.Tensor([t_enc] * int(img.shape[0]))

        x = self.sampler.stochastic_encode(img, t, noise=noise)
        
        samples = self.sampler.decode(x, conditioning, t_enc, 
                                        unconditional_guidance_scale=self.scale,
                                        unconditional_conditioning=unconditional_conditioning)
        return samples


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value