import gc
from typing import Callable
from collections import Ordereddict

import torch

module_in_gpu = None
cpu = torch.device("cpu")
gpu = torch.device("cuda")
cache = {}
device = gpu if torch.cuda.is_available() else cpu
device_lookup = {
    "c": cpu,
    "cpu": cpu,
    "g": gpu,
    "gpu": gpu,
    "cuda": gpu,
    "device": device
}

def clear_cuda(show_summary=False):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if show_summary:
        torch.cuda.memory_summary(device=None, abbreviated=True)

def torch_gc():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def remove_forward_pre_hook(model: torch.nn.Module) -> None:
    model._forward_pre_hooks = Ordereddict()

def remove_all_forward_pre_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_pre_hooks"):
                remove_forward_pre_hook(child)
            remove_all_forward_pre_hooks(child)

def setup_for_grads(model):        
    vae = model.first_stage_model
    unet = model.model.diffusion_model

    remove_forward_pre_hook(vae)
    remove_all_forward_pre_hooks(unet)

def gpu_swap(model_dict):
    parents = {}
    def send_me_to_gpu(module, _):
        """send this module to GPU; send whatever tracked module was previous in GPU to CPU;
        we add this as forward_pre_hook to a lot of modules and this way all but one of them will
        be in CPU
        """
        global module_in_gpu

        if module_in_gpu is not None:
            module_in_gpu.to(cpu)

        module.to(gpu)
        module_in_gpu = module
    def encode_wrap(self, encode, x):
        send_me_to_gpu(self, None)
        return encode(x)
    def decode_wrap(self, decode, z):
        send_me_to_gpu(self, None)
        return decode(z)
    vae = model_dict["vae"]
    unet = model_dict["unet"]
    cache["encode"] = vae.encode
    cache["decode"] = vae.decode
    model_dict["vae"].encode = lambda x, en=vae.encode: encode_wrap(vae, en, x)
    model_dict["vae"].decode = lambda x, de=vae.decode: decode_wrap(vae, de, x)
    
def setup_for_low_vram(sd_model, use_medvram=False):
    parents = {}

    def send_me_to_gpu(module, _):
        """send this module to GPU; send whatever tracked module was previous in GPU to CPU;
        we add this as forward_pre_hook to a lot of modules and this way all but one of them will
        be in CPU
        """
        global module_in_gpu

        module = parents.get(module, module)

        if module_in_gpu == module:
            return

        if module_in_gpu is not None:
            module_in_gpu.to(cpu)

        module.to(gpu)
        module_in_gpu = module

    # see below for register_forward_pre_hook;
    # first_stage_model does not use forward(), it uses encode/decode, so register_forward_pre_hook is
    # useless here, and we just replace those methods
    def first_stage_model_encode_wrap(self, encoder, x):
        send_me_to_gpu(self, None)
        return encoder(x)

    def first_stage_model_decode_wrap(self, decoder, z):
        send_me_to_gpu(self, None)
        return decoder(z)

    # remove three big modules, cond, first_stage, and unet from the model and then
    # send the model to GPU. Then put modules back. the modules will be in CPU.
    if hasattr(sd_model.cond_stage_model, "transformer"):
        stored = sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.model
        sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.model = None, None, None
        sd_model.to(device)
        sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.model = stored
    else:
        stored = sd_model.cond_stage_model, sd_model.first_stage_model, sd_model.model
        sd_model.cond_stage_model, sd_model.first_stage_model, sd_model.model = None, None, None
        sd_model.to(device)
        sd_model.cond_stage_model, sd_model.first_stage_model, sd_model.model = stored

    # register hooks for those the first two models
    if hasattr(sd_model.cond_stage_model, "transformer"):
        sd_model.cond_stage_model.transformer.register_forward_pre_hook(send_me_to_gpu)
    else:
        sd_model.cond_stage_model.register_forward_pre_hook(send_me_to_gpu)
    sd_model.first_stage_model.register_forward_pre_hook(send_me_to_gpu)
    cache["encode"] = sd_model.first_stage_model.encode
    cache["decode"] = sd_model.first_stage_model.decode
    sd_model.first_stage_model.encode = lambda x, en=sd_model.first_stage_model.encode: first_stage_model_encode_wrap(sd_model.first_stage_model, en, x)
    sd_model.first_stage_model.decode = lambda z, de=sd_model.first_stage_model.decode: first_stage_model_decode_wrap(sd_model.first_stage_model, de, z)
    if hasattr(sd_model.cond_stage_model, "transformer"):
        parents[sd_model.cond_stage_model.transformer] = sd_model.cond_stage_model

    if use_medvram:
        sd_model.model.register_forward_pre_hook(send_me_to_gpu)
    else:
        diff_model = sd_model.model.diffusion_model

        # the third remaining model is still too big for 4 GB, so we also do the same for its submodules
        # so that only one of them is in GPU at a time
        stored = diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = None, None, None, None
        diff_model.to(device)
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = stored

        # install hooks for bits of third model
        diff_model.time_embed.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.input_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)
        diff_model.middle_block.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.output_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)
