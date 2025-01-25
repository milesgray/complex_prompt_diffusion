import random
from functools import partial
from collections import defaultdict
import numpy as np
import torch
import torchvision.transforms as K
import skimage
from tqdm import trange

from cpd.util import Map

class NoiseGenerator:
    def __init__(
        self, 
        shape, 
        device, 
        seed=0, 
        torch_generator=None, 
        seed_mode="iter", 
        cycle_size=5, 
        logger=print
    ):
        self._log = logger
        self._seed = seed
        self.seed_mode = seed_mode
        self.generator = torch_generator
        self.fn = partial(torch.randn, shape, device=device)
        self._seed_list = build_cycle_mod(n=cycle_size)
        self._seed_idx = 0
        self._seed_lookup = {}
        self._exemplar_sequence = {}
        self._exemplars = defaultdict(list)

    @property
    def seed(self, key=None):
        if key in self._seed_lookup:
            return self._seed_lookup[key]
        if self.seed_mode == "iter":
            self._seed += 1
        elif self.seed_mode in ["constant", "const", "c"]:
            self._seed = self._seed
        elif self.seed_mode in ["loop", "l"]:
            self._seed = self._seed_list[self._seed_idx % len(self._seed_list)]
        else:
            self._seed = random.randint(0,10000)
        return self._seed
    
    @property
    def last_seed(self):
        return self._seed

    def add_exemplar(self, value, seed, uid, name=""):
        if uid not in self._exemplar_sequence:
            self._exemplar_sequence[uid] = 0
        self._exemplars[uid].append(Map({
            "value": value,
            "seed": seed,            
            "name": name,
        }))
        return len(self._exemplars[uid])
    
    def clear_exemplars(self, uid):
        if uid in self._exemplar_sequence:
            self._exemplar_sequence.pop(uid)
        if uid in self._exemplars:
            self._exemplars.pop(uid)

    def reset_sequence(self, uid):
        assert uid in self._exemplar_sequence, f"UID sequence not found, add an exemplar under this UID first: {uid}"
        self._exemplar_sequence[uid] = 0

    def sample_sequence(self, uid, match=False, verbose=False):
        assert uid in self._exemplars, f"UID not found, add an exemplar under this UID first: {uid}"
        assert uid in self._exemplar_sequence, f"UID sequence not found, add an exemplar under this UID first: {uid}"
        exemplar = self._exemplars[uid][self._exemplar_sequence[uid]]
        if match:
            result = self.sample(match_noise=exemplar.value)
        else:
            result = self.sample(seed=exemplar.seed)
        self._exemplar_sequence[uid] += 1        
        if self._exemplar_sequence[uid] >= len(self._exemplars[uid]):
            self._exemplar_sequence[uid] = 0
        if verbose: self._log(f"[sample_sequence]\tGot noise for step {exemplar.name} using seed {exemplar.seed} - result: {result.mean():0.5f}, exemplar: {exemplar.value.mean():0.5f}")
        return result

    def sample(self, seed=None, match_noise=None):
        if seed is None:
            seed = self.seed
        torch.manual_seed(seed)
        result = self.fn()
        if match_noise is not None:
            result = torch.from_numpy(skimage.exposure.match_histograms(result.numpy(), match_noise.numpy(), multichannel=True))
        return result

def build_cycle_mod(n=5):
    return [x for x in range(1,n)] + [-x for x in range(1,n)][::-1]


def _fft2(data):
    if data.ndim > 2:  # has channels
        out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:, :, c]
            out_fft[:, :, c] = np.fft.fft2(np.fft.fftshift(c_data), norm="ortho")
            out_fft[:, :, c] = np.fft.ifftshift(out_fft[:, :, c])
    else:  # one channel
        out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_fft[:, :] = np.fft.fft2(np.fft.fftshift(data), norm="ortho")
        out_fft[:, :] = np.fft.ifftshift(out_fft[:, :])

    return out_fft


def _ifft2(data):
    if data.ndim > 2:  # has channels
        out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:, :, c]
            out_ifft[:, :, c] = np.fft.ifft2(np.fft.fftshift(c_data), norm="ortho")
            out_ifft[:, :, c] = np.fft.ifftshift(out_ifft[:, :, c])
    else:  # one channel
        out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_ifft[:, :] = np.fft.ifft2(np.fft.fftshift(data), norm="ortho")
        out_ifft[:, :] = np.fft.ifftshift(out_ifft[:, :])

    return out_ifft


def _get_gaussian_window(width, height, std=3.14, mode=0):
    window_scale_x = float(width / min(width, height))
    window_scale_y = float(height / min(width, height))

    window = np.zeros((width, height))
    x = (np.arange(width) / width * 2. - 1.) * window_scale_x
    for y in range(height):
        fy = (y / height * 2. - 1.) * window_scale_y
        if mode == 0:
            window[:, y] = np.exp(-(x ** 2 + fy ** 2) * std)
        else:
            window[:, y] = (1 / ((x ** 2 + 1.) * (fy ** 2 + 1.))) ** (
                    std / 3.14)  # hey wait a minute that's not gaussian

    return window


def _get_masked_window_rgb(np_mask_grey, hardness=1.):
    np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
    if hardness != 1.:
        hardened = np_mask_grey[:] ** hardness
    else:
        hardened = np_mask_grey[:]
    for c in range(3):
        np_mask_rgb[:, :, c] = hardened[:]
    return np_mask_rgb


def get_matched_noise(_np_src_image, np_mask_rgb, noise_q, color_variation):
    global DEBUG_MODE
    global TMP_ROOT_PATH

    width = _np_src_image.shape[0]
    height = _np_src_image.shape[1]
    num_channels = _np_src_image.shape[2]

    np_src_image = _np_src_image[:] * (1. - np_mask_rgb)
    np_mask_grey = (np.sum(np_mask_rgb, axis=2) / 3.)
    np_src_grey = (np.sum(np_src_image, axis=2) / 3.)
    all_mask = np.ones((width, height), dtype=bool)
    img_mask = np_mask_grey > 1e-6
    ref_mask = np_mask_grey < 1e-3

    windowed_image = _np_src_image * (1. - _get_masked_window_rgb(np_mask_grey))
    windowed_image /= np.max(windowed_image)
    windowed_image += np.average(
        _np_src_image) * np_mask_rgb  # / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black, we get better results from fft by filling the average unmasked color
    # windowed_image += np.average(_np_src_image) * (np_mask_rgb * (1.- np_mask_rgb)) / (1.-np.average(np_mask_rgb)) # compensate for darkening across the mask transition area
    # _save_debug_img(windowed_image, "windowed_src_img")

    src_fft = _fft2(windowed_image)  # get feature statistics from masked src img
    src_dist = np.absolute(src_fft)
    src_phase = src_fft / src_dist
    # _save_debug_img(src_dist, "windowed_src_dist")

    noise_window = _get_gaussian_window(width, height, mode=1)  # start with simple gaussian noise
    noise_rgb = np.random.random_sample((width, height, num_channels))
    noise_grey = (np.sum(noise_rgb, axis=2) / 3.)
    noise_rgb *= color_variation  # the colorfulness of the starting noise is blended to greyscale with a parameter
    for c in range(num_channels):
        noise_rgb[:, :, c] += (1. - color_variation) * noise_grey

    noise_fft = _fft2(noise_rgb)
    for c in range(num_channels):
        noise_fft[:, :, c] *= noise_window
    noise_rgb = np.real(_ifft2(noise_fft))
    shaped_noise_fft = _fft2(noise_rgb)
    shaped_noise_fft[:, :, :] = np.absolute(shaped_noise_fft[:, :, :]) ** 2 * (
            src_dist ** noise_q) * src_phase  # perform the actual shaping

    brightness_variation = 0.  # color_variation # todo: temporarily tieing brightness variation to color variation for now
    contrast_adjusted_np_src = _np_src_image[:] * (brightness_variation + 1.) - brightness_variation * 2.

    # scikit-image is used for histogram matching, very convenient!
    shaped_noise = np.real(_ifft2(shaped_noise_fft))
    shaped_noise -= np.min(shaped_noise)
    shaped_noise /= np.max(shaped_noise)
    shaped_noise[img_mask, :] = skimage.exposure.match_histograms(shaped_noise[img_mask, :] ** 1.,
                                                                  contrast_adjusted_np_src[ref_mask, :], multichannel=1)
    shaped_noise = _np_src_image[:] * (1. - np_mask_rgb) + shaped_noise * np_mask_rgb
    # _save_debug_img(shaped_noise, "shaped_noise")

    matched_noise = np.zeros((width, height, num_channels))
    matched_noise = shaped_noise[:]
    # matched_noise[all_mask,:] = skimage.exposure.match_histograms(shaped_noise[all_mask,:], _np_src_image[ref_mask,:], channel_axis=1)
    # matched_noise = _np_src_image[:] * (1. - np_mask_rgb) + matched_noise * np_mask_rgb

    # _save_debug_img(matched_noise, "matched_noise")

    """
	todo:
	color_variation doesnt have to be a single number, the overall color tone of the out-painted area could be param controlled
	"""

    return np.clip(matched_noise, 0., 1.)

def find_noise_for_image(model, device, init_image, prompt, steps=200, cond_scale=2.0, verbose=False, normalize=False,
                         generation_callback=None):
    image = np.array(init_image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2. * image - 1.
    image = image.to(device)
    x = model.get_first_stage_encoding(model.encode_first_stage(image))

    uncond = model.get_learned_conditioning([''])
    cond = model.get_learned_conditioning([prompt])

    s_in = x.new_ones([x.shape[0]])
    dnw = K.external.CompVisDenoiser(model)
    sigmas = dnw.get_sigmas(steps).flip(0)

    if verbose:
        print(sigmas)

    for i in trange(1, len(sigmas)):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
        cond_in = torch.cat([uncond, cond])

        c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]

        if i == 1:
            t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
        else:
            t = dnw.sigma_to_t(sigma_in)

        eps = model.apply_model(x_in * c_in, t, cond=cond_in)
        denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

        denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale

        if i == 1:
            d = (x - denoised) / (2 * sigmas[i])
        else:
            d = (x - denoised) / sigmas[i - 1]

        if generation_callback is not None:
            generation_callback(x, i)

        dt = sigmas[i] - sigmas[i - 1]
        x = x + d * dt

    return x / sigmas[-1]    