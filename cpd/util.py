import warnings
import random
import importlib
from pathlib import Path
from inspect import isfunction
from collections import OrderedDict, defaultdict
from typing import Dict, Callable


import PIL
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as tfms
import matplotlib.pyplot as plt

import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import skimage.measure

from cpd.vram import device_lookup

# miscellaneous function for reading, writing and processing rgb and depth images.
def resizewithpool(img, size):
    i_size = img.shape[0]
    n = int(np.floor(i_size/size))

    out = skimage.measure.block_reduce(img, (n, n), np.max)
    return out

def showimage(img):
    plt.imshow(img)
    plt.colorbar()
    plt.show()

def read_image(path):
    img = cv2.imread(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img

def generatemask(size):
    # Generates a Guassian mask
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0]/16)
    k_size = int(2 * np.ceil(2 * int(size[0]/16)) + 1)
    mask[int(0.15*size[0]):size[0] - int(0.15*size[0]), int(0.15*size[1]): size[1] - int(0.15*size[1])] = 1
    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask

def impatch(image, rect):
    # Extract the given patch pixels from a given image.
    w1 = rect[0]
    h1 = rect[1]
    w2 = w1 + rect[2]
    h2 = h1 + rect[3]
    image_patch = image[h1:h2, w1:w2]
    return image_patch

def getGF_fromintegral(integralimage, rect):
    # Computes the gradient density of a given patch from the gradient integral image.
    x1 = rect[1]
    x2 = rect[1]+rect[3]
    y1 = rect[0]
    y2 = rect[0]+rect[2]
    value = integralimage[x2, y2]-integralimage[x1, y2]-integralimage[x2, y1]+integralimage[x1, y1]
    return value

def rgb2gray(rgb):
    # Converts rgb to gray
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def calculateprocessingres(img, basesize, confidence=0.1, scale_threshold=3, whole_size_threshold=3000):
    # Returns the R_x resolution described in section 5 of the main paper.

    # Parameters:
    #    img :input rgb image
    #    basesize : size the dilation kernel which is equal to receptive field of the network.
    #    confidence: value of x in R_x; allowed percentage of pixels that are not getting any contextual cue.
    #    scale_threshold: maximum allowed upscaling on the input image ; it has been set to 3.
    #    whole_size_threshold: maximum allowed resolution. (R_max from section 6 of the main paper)

    # Returns:
    #    outputsize_scale*speed_scale :The computed R_x resolution
    #    patch_scale: K parameter from section 6 of the paper

    # speed scale parameter is to process every image in a smaller size to accelerate the R_x resolution search
    speed_scale = 32
    image_dim = int(min(img.shape[0:2]))

    gray = rgb2gray(img)
    grad = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)) + np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    grad = cv2.resize(grad, (image_dim, image_dim), cv2.INTER_AREA)

    # thresholding the gradient map to generate the edge-map as a proxy of the contextual cues
    m = grad.min()
    M = grad.max()
    middle = m + (0.4 * (M - m))
    grad[grad < middle] = 0
    grad[grad >= middle] = 1

    # dilation kernel with size of the receptive field
    kernel = np.ones((int(basesize/speed_scale), int(basesize/speed_scale)), np.float)
    # dilation kernel with size of the a quarter of receptive field used to compute k
    # as described in section 6 of main paper
    kernel2 = np.ones((int(basesize / (4*speed_scale)), int(basesize / (4*speed_scale))), np.float)

    # Output resolution limit set by the whole_size_threshold and scale_threshold.
    threshold = min(whole_size_threshold, scale_threshold * max(img.shape[:2]))

    outputsize_scale = basesize / speed_scale
    for p_size in range(int(basesize/speed_scale), int(threshold/speed_scale), int(basesize / (2*speed_scale))):
        grad_resized = resizewithpool(grad, p_size)
        grad_resized = cv2.resize(grad_resized, (p_size, p_size), cv2.INTER_NEAREST)
        grad_resized[grad_resized >= 0.5] = 1
        grad_resized[grad_resized < 0.5] = 0

        dilated = cv2.dilate(grad_resized, kernel, iterations=1)
        meanvalue = (1-dilated).mean()
        if meanvalue > confidence:
            break
        else:
            outputsize_scale = p_size

    grad_region = cv2.dilate(grad_resized, kernel2, iterations=1)
    patch_scale = grad_region.mean()

    return int(outputsize_scale*speed_scale), patch_scale

def applyGridpatch(blsize, stride, img, box):
    # Extract a simple grid patch.
    counter1 = 0
    patch_bound_list = {}
    for k in range(blsize, img.shape[1] - blsize, stride):
        for j in range(blsize, img.shape[0] - blsize, stride):
            patch_bound_list[str(counter1)] = {}
            patchbounds = [j - blsize, k - blsize, j - blsize + 2 * blsize, k - blsize + 2 * blsize]
            patch_bound = [box[0] + patchbounds[1], box[1] + patchbounds[0], patchbounds[3] - patchbounds[1],
                           patchbounds[2] - patchbounds[0]]
            patch_bound_list[str(counter1)]['rect'] = patch_bound
            patch_bound_list[str(counter1)]['size'] = patch_bound[2]
            counter1 = counter1 + 1
    return patch_bound_list

class Images:
    def __init__(self, root_dir, files, index):
        self.root_dir = root_dir
        name = files[index]
        self.rgb_image = read_image(os.path.join(self.root_dir, name))
        name = name.replace(".jpg", "")
        name = name.replace(".png", "")
        name = name.replace(".jpeg", "")
        self.name = name

class ImageandPatchs:
    def __init__(self, root_dir, name, patchsinfo, rgb_image, scale=1):
        self.root_dir = root_dir
        self.patchsinfo = patchsinfo
        self.name = name
        self.patchs = patchsinfo
        self.scale = scale

        self.rgb_image = cv2.resize(rgb_image, (round(rgb_image.shape[1]*scale), round(rgb_image.shape[0]*scale)),
                                    interpolation=cv2.INTER_CUBIC)

        self.do_have_estimate = False
        self.estimation_updated_image = None
        self.estimation_base_image = None

    def __len__(self):
        return len(self.patchs)

    def set_base_estimate(self, est):
        self.estimation_base_image = est
        if self.estimation_updated_image is not None:
            self.do_have_estimate = True

    def set_updated_estimate(self, est):
        self.estimation_updated_image = est
        if self.estimation_base_image is not None:
            self.do_have_estimate = True

    def __getitem__(self, index):
        patch_id = int(self.patchs[index][0])
        rect = np.array(self.patchs[index][1]['rect'])
        msize = self.patchs[index][1]['size']

        ## applying scale to rect:
        rect = np.round(rect * self.scale)
        rect = rect.astype('int')
        msize = round(msize * self.scale)

        patch_rgb = impatch(self.rgb_image, rect)
        if self.do_have_estimate:
            patch_whole_estimate_base = impatch(self.estimation_base_image, rect)
            patch_whole_estimate_updated = impatch(self.estimation_updated_image, rect)
            return {'patch_rgb': patch_rgb, 'patch_whole_estimate_base': patch_whole_estimate_base,
                    'patch_whole_estimate_updated': patch_whole_estimate_updated, 'rect': rect,
                    'size': msize, 'id': patch_id}
        else:
            return {'patch_rgb': patch_rgb, 'rect': rect, 'size': msize, 'id': patch_id}

class ImageDataset:
    def __init__(self, root_dir, subsetname):
        self.dataset_dir = root_dir
        self.subsetname = subsetname
        self.rgb_image_dir = root_dir
        self.files = sorted(os.listdir(self.rgb_image_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return Images(self.rgb_image_dir, self.files, index)

def randn_tensor(
    shape,
    generator,
    device=None,
    dtype=None,
    layout=None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

def pil_to_latent(vae, input_im, device, generator=None):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(device)*2-1) # Note scaling
    latent = 0.18215 * latent.latent_dist.sample(generator)
    
    return latent

def latents_to_pil(vae, latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def latents_to_img(vae, latents, normalize='0-1', size=512):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    # with torch.no_grad():
    image = vae.decode(latents).sample
    if normalize == '0-1':
        image = (image / 2 + 0.5)
    # resize tensor to 224x224 
    image = torch.nn.functional.interpolate(image, size=(size, size), mode='bilinear', align_corners=False)
    return image

def img_to_latents(vae, input_im, generator=None):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    latent = vae.encode(input_im)
    latent = 0.18215 * latent.latent_dist.sample(generator)
    
    return latent

def get_timesteps(scheduler, num_inference_steps, strength):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start

def prepare_latents(vae, scheduler, batch_size, num_channels_latents, height, width, device, generator=None):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    latents = randn_tensor(shape, generator=generator, device=device)
    latents = latents * scheduler.init_noise_sigma
    
    return latents

def encode_text(prompt, tokenizer, text_encoder, device):
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    text_embeddings = text_embeddings.clone()
    
    return text_embeddings

def visualize_latents(latents, title='Image'):
    model_output_img = latents_to_pil(latents)
    plt.imshow(model_output_img[0])
    plt.title(title)
    plt.show()
    
# define a function that visualize a list of pil images
def visualize_images(images, titles=None, cols=5, figsize=(3, 3)):   
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if n_images == 1:
        images[0].show()
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure(figsize=figsize)
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(int(np.ceil(n_images/float(cols))), int(cols), n + 1)
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    

def load_mask(path, size=(64,64)):
    mask = np.array(load_image(path))
    return shape_mask(mask)

def shape_mask(mask, size=(64,64)):
    mask[mask > 50] = 255
    mask[mask < 50] = 0
    mask = torch.Tensor(mask)[:,:,0].unsqueeze(0).unsqueeze(0) / 255
    mask = torch.nn.functional.interpolate(mask, size=size, mode='nearest', recompute_scale_factor=False)
    mask = torch.asarray(mask).type(torch.float16)
    nmask = torch.asarray(1.0 - mask).type(torch.float16)
    return mask

def save_image(image, path, force_name=None, return_path=False):
    count = len([_ for _ in path.glob("*.png")])
    if force_name:
        save_path = path / f"{force_name}.png"
    else:
        save_path = path / f"{count}.png"
    image.save(save_path)
    if return_path:
        return save_path
    else:
        return image

def load_image(path):
    img_exists = Path(path).exists()
    if img_exists:
        loaded_img = PIL.Image.open(path)
        if not loaded_img.mode == "RGB":
            loaded_img = loaded_img.convert("RGB")
        return loaded_img
    else:
        return None
    
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())
    
def to_pil(x):
    return PIL.Image.fromarray(x, 'RGB')

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def pil_to_numpy(img):
    return np.array(img).astype(np.uint8)
def pil_to_torch(img):
    return torch.from_numpy(pil_to_numpy(img)) \
        .div(255) \
            .mul(2). \
                subtract(1) \
                    .permute(2,0,1)

def safe_to(x, device=None, dtype=None, clone=False):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif isinstance(x, (int, float)):
        x = torch.Tensor([x])
    elif isinstance(x, (list, tuple)):
        x = type(x)([safe_to(t, device=device, dtype=dtype) for t in x])
    elif isinstance(x, (dict, defaultdict, OrderedDict, Map)):
        x = {k: safe_to(v, device=device, dtype=dtype) for k,v in x.items()}
    if isinstance(x, (torch.Tensor, torch.nn.Module)):
        if clone:
            x = x.clone()
        if device:
            if isinstance(device, str):
                if device in device_lookup:
                    device = device_lookup[device]
                else:
                    device = torch.device(device)
            x = x.to(device)
        if dtype:
            if isinstance(dtype, str):
                if dtype in torch.__dict__:
                    dtype = torch.__dict__[dtype] 
                else:
                    raise ValueError(f"Unknown dtype '{dtype}'")               
            x = x.to(dtype)
    return x

class Map(dict):
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


class CudaMon:
    def __init__(self, class_name, log_fn=print, verbose=False):
        self.class_name = class_name
        self.log = log_fn
        self.verbose = verbose

    def __call__(self, method, action):
        if self.verbose:
            self.log_fn(f"[{self.class_name}.{method}]\t[{action}]\t-\t{torch.cuda.memory_allocated()}")

def seed_everything(seed, cudnn_deterministic=False, verbose=True):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    
    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
        if verbose: print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)

def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params

def from_json(json, **kwargs):
    return get_obj_from_str(f"{json['module']}.{json['class']}").from_json(json, **kwargs)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

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