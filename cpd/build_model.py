#@markdown #Get the inference model directly from huggingface.co
!pip install omegaconf
import os

STABLE_DIFFUSION_MODEL = "512-base-ema.ckpt" #@param ["768-v-ema.ckpt", "512-base-ema.ckpt"]
DOWNLOAD_INPAINTING_MODEL = False #@param {type: "boolean"}
IS_BASE_MODEL = STABLE_DIFFUSION_MODEL == '512-base-ema.ckpt'
MODEL_URL = None
STABLE_DIFFUSION_MODEL_INPAINTING = "512-inpainting-ema.ckpt"


if not IS_BASE_MODEL:
  MODEL_URL = "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt"
else:
  MODEL_URL = "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt"

if not MODEL_URL == None and not os.path.exists(STABLE_DIFFUSION_MODEL):
    !wget $MODEL_URL
else:
    print("Skip model download.")

if DOWNLOAD_INPAINTING_MODEL and not os.path.exists(STABLE_DIFFUSION_MODEL_INPAINTING):
    !wget "https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/$STABLE_DIFFUSION_MODEL_INPAINTING"
else:
    print("Skip inpainting model download.")

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

import pytorch_lightning
from omegaconf import OmegaConf
import torch
import sys
add_paths =["/deps/taming-transformers", 
            "/content/stablediffusion",
            "/content/sg_container"]
for p in add_paths:
  if p not in sys.path:
    sys.path.append(p)

from ldm.util import instantiate_from_config
config = OmegaConf.load(f"/content/sg_container/v2-inference.yaml")
model = load_model_from_config(config, f"/content/v2-1_512-ema-pruned.ckpt")