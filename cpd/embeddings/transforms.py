import os, json
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Union, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Union, Tuple, List, Callable, Dict

import numpy as np
from PIL import Image
from tqdm.notebook import tqdm

import torch
import torch.nn.functional as nnf
from torch.optim.adam import Adam

from cpd.scheduler.ddim import DDIMScheduler

from cpd.util import from_json
from cpd.embeddings.prompts import ComplexPrompt

@torch.no_grad()
def bleed(x: Tensor) -> Tensor:
    def make_filter() -> nn.Conv2d:
        conv = nn.Conv2d(1, 1, 14, 1, 7, bias=False)
        f = np.array([[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.050, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.100, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.200, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.250, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.300, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.400, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]])
        conv.weight = nn.Parameter(torch.from_numpy(f).unsqueeze(0).double())
        return conv.to(x.device)
    return make_filter()(x.view(1,1,x.shape[-2], x.shape[-1])).view(x.shape)


def plerp(xp: Tensor, yp: Tensor, x: float, threshold: float):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1, N = 1 for Transform).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    x = Tensor([x])
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def slerp(v0: Tensor, v1: Tensor, t: float, threshold: float) -> Tensor:
    """
    Spherical Linear Interpolation between two n-dimensional features.

    Args:
        v0 (Tensor): N-dimensional feature to start from.
            When `t` is 0.0, result is equal to `v0`.
        v1 (Tensor): N-dimensional feature to interpolate to.
            When `t` is 1.0, result is equal to `v1`.
        t (float): Value between 0 and 1 indicating degree of
            interpolation between `v0` and `v1`. As `t` increases,
            result becomes closer to `v1`.
        threshold (float): When the dot product between the two
            features is greater than `threshold`, fall back to
            linear interpolation.

    Returns:
        Tensor: Interpolated value, shape matches `v0` and `v1`
    """
    device = v0.device
    
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()
    original_range = (min(v0.min(), v1.min()), max(v0.max(),v1.max()))
    
    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > threshold:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1
    v2 = np.clip(v2, original_range[0], original_range[1])
    return torch.from_numpy(v2).to(device)

def lerp(v0: Tensor, v1: Tensor, t: float, threshold: float) -> Tensor:
    """
    Linear Interpolation between two n-dimensional features.

    Args:
        v0 (Tensor): N-dimensional feature to start from.
            When `t` is 0.0, result is equal to `v0`.
        v1 (Tensor): N-dimensional feature to interpolate to.
            When `t` is 1.0, result is equal to `v1`.
        t (float): Value between 0 and 1 indicating degree of
            interpolation between `v0` and `v1`. As `t` increases,
            result becomes closer to `v1`.
        threshold (float): Not used.

    Returns:
        Tensor: Interpolated value, shape matches `v0` and `v1`
    """
    device = v0.device
    
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()
    original_range = (min(v0.min(), v1.min()), max(v0.max(),v1.max()))
    v0 = (1 - t) * v0
    v1 = t * v1
    v2 = v0 + v1
    v2 = np.clip(v2, original_range[0], original_range[1])
    return torch.from_numpy(v2).to(device)

interpolate_lookup = {
    "plerp": plerp,
    "slerp": slerp,
    "lerp": lerp,
}

def valid_range(S: int, r: tuple=None, idxs: list=None) -> dict:
    """
    Create a valid start and end value for a given size or list of indexes.
    Used to sanity check input and silently fix any invalid range provided.

    Args:
        S (int): size of the tensor the range is made for
        r (tuple, optional): a tuple of start/end values to be used. 
        Defaults to None, in which case a range of 0,`S` will be used.
        idxs (list, optional): A list of non-consecutive indexes. Will be used
        to determine valid min/max values of the range. Defaults to None,
        in which case the min will be 0 and the max will be `S`.

    Returns:
        dict: "start" and "end" values that are valid given the inputs.
    """
    r = (0,S) if r is None else r
    r_min = 0 if idxs is None else min(idxs)
    r_max = S if idxs is None else max(idxs)        
    r_start = max(min(r[0], r[1]), r_min)
    r_end = min(max(r[0], r[1]), r_max)
    return {
        "start": r_start, 
        "end": r_end
    }

class AbstractTransform:
    def __init__(self, args: dict):
        self.args = args
        self.param_lerp_keys = args['lerp_keys'] if 'lerp_keys' in args else []
        self.step_results = []

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json(self) -> Dict[str, Any]:
        """
        Packs this instance to a JSON dictionary that can be serialized.
        All values are strings or containers with strings.

        Returns:
            `dict`: Serializable dictionary containing all the attributes that make up this instance.
        """
        type_str = str(type(self)).replace("<class '","").replace("'>","")
        module_str = ".".join(type_str.split(".")[:-1])
        json = {        
            "args": self.args,
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
        }      
        return json

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """        
        return json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    @classmethod
    def from_json(cls, json: dict, **kwargs):
        """
        Instantiates an object of this type from the given dictionary.
        The dictionary is expected to be the output from `to_json` and
        together allows for easy deserialization/serialization.

        Args:
            json (dict): JSON compatible dictionary containing all values needed
            to reinstantiate a serialized instance.

        Returns:
            `cls`: Deserialized instance
        """
        return cls(json["args"]) 

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)
    
    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike], **kwargs):
        json = cls._dict_from_json_file(json_file)
        return cls.from_json(json, **kwargs)

class AbstractPromptTransform(AbstractTransform):
    def __init__(self, target: ComplexPrompt, args: dict):
        super().__init__(args)
        self.target = target
        self.cache = OrderedDict()
        
    def to_json(self) -> Dict[str, Any]:
        json = super().to_json()
        json["target"] = self.target.to_json()
        return json

    @classmethod
    def from_json(cls, json: dict, **kwargs):
        """
        Instantiates an object of this type from the given dictionary.
        The dictionary is expected to be the output from `to_json` and
        together allows for easy deserialization/serialization.

        Args:
            json (dict): JSON compatible dictionary containing all values needed
            to reinstantiate a serialized instance.

        Returns:
            `cls`: Deserialized instance
        """
        return cls(from_json(json["target"], **kwargs), json["args"]) 
        
    def apply(self, prompt_start, steps=1, verbose=False): 
        if len(self.param_lerp_keys) == 0 or \
            all([k not in self.args for k in self.param_lerp_keys]):
            steps = 1
            if verbose:
                print(f"No valid interpolatable parameters found. Set them as values for 'lerp_keys' in config dictionary")    
        for s in range(max(1,steps)):
            if verbose:
                print(f"[{s+1}/{steps}] {(s+1)/steps}%")
            params = self.lerp_params(self.args, (s+1)/steps, verbose=verbose)
            batch_embedding = self.step(prompt_start, self.target, params, 
                                        verbose=verbose)
            self.step_results.append(batch_embedding)
        return self.step_results

    def step(self, prompt_start, prompt_end, params, verbose=False):
        raise NotImplementedError

    def lerp_params(self, params, amount, verbose=False):
        if amount == 1:
            return params
        result = {}        
        for k,v in params.items():
            if k not in self.param_lerp_keys:
                result[k] = v
            else:
                if isinstance(v, float):
                    result[k] = v * amount
                    if verbose:
                        print(f"[{k}] {v} -> {result[k]}")
                elif isinstance(v, int):
                    result[k] = int(v * amount)
                    if verbose:
                        print(f"[{k}] {v} -> {result[k]}")
                elif isinstance(v, tuple):
                    if len(v) != 2: 
                        result[k] = v
                    elif isinstance(v[0], int) and isinstance(v[1], int):
                        v[0] = int(v[0] * amount)
                        v[1] = int(v[1] + v[1] * (1-amount))
                        result[k] = v
                    elif isinstance(v[0], float) and isinstance(v[1], float):
                        v[0] = v[0] * amount
                        v[1] = v[1] + v[1] * (1-amount)
                        result[k] = v
                    else:
                        result[k] = v
                    if verbose:
                        print(f"[{k}] {v} -> {result[k]}")
                else:
                    result[k] = v
                    if verbose:
                        print(f"[{k}] {v} -> {result[k]}")
        if verbose:
            print(f"Result: {result}")
        return result


class LerpCLIPEmbeddingTransform(AbstractPromptTransform):
    def __init__(self, target: ComplexPrompt, args: dict):
        super().__init__(target, args)

        self.args['magnitude'] = self.args['magnitude'] \
                if 'magnitude' in self.args else 1.0
        self.args['lerp_threshold'] = self.args['lerp_threshold'] \
                if 'lerp_threshold' in self.args else 0.995
        self.args['lerp_mode'] = self.args['lerp_mode'] \
                if 'lerp_mode' in self.args else "slerp"
        self.args['do_bleed'] = self.args['do_bleed'] \
                if 'do_bleed' in self.args else False

        self.interp = interpolate_lookup[self.args['lerp_mode']] \
                if self.args['lerp_mode'] in interpolate_lookup else slerp

        self.args['token_k'] = self.args['token_k'] \
                if 'token_k' in self.args else 77
        self.args['token_idxs'] = self.args['token_idxs'] \
                if 'token_idxs' in self.args else None
        self.args['token_range'] = self.args['token_range'] \
                if 'token_range' in self.args else None        
        self.args['token_largest'] = self.args['token_largest'] \
                if 'token_largest' in self.args else True

        
        self.args['embed_k'] = self.args['embed_k'] \
                if 'embed_k' in self.args else 768
        self.args['embed_idxs'] = self.args['embed_idxs'] \
                if 'embed_idxs' in self.args else None
        self.args['embed_range'] = self.args['embed_range'] \
                if 'embed_range' in self.args else None        
        self.args['embed_largest'] = self.args['embed_largest'] \
                if 'embed_largest' in self.args else True

        self.args['delta_mult'] = self.args['delta_mult'] \
                if 'delta_mult' in self.args else 1.0
        self.args['static_mult'] = self.args['static_mult'] \
                if 'static_mult' in self.args else 1.0

    def step(self, prompt_start: ComplexPrompt, prompt_end: ComplexPrompt, params: dict, verbose=False) -> Tensor:
        if "start" not in self.cache:
            c_start = list([e for e in prompt_start.get_embeddings(force=True)])        
            c_start = [torch.from_numpy(c) if isinstance(c, np.ndarray) else c for c in c_start]
            c_start = [c.cpu().double() for c in c_start]
            self.cache["start"] = c_start
        else:
            c_start = self.cache["start"]
        
        if "end" not in self.cache:
            c_end = [e for e in prompt_end.get_embeddings()]
            c_end = [torch.from_numpy(c) if isinstance(c, np.ndarray) else c for c in c_end]
            c_end = [c.cpu().double() for c in c_end]
            self.cache["end"] = c_end
        else:
            c_end = self.cache["end"]
        token_maps = (prompt_start.token_map, prompt_end.token_map)

        assert all((cs.shape == ce.shape for cs,ce in zip(c_start, c_end)))

        batch_size = 1

        results = []
        results_guide = []
        for b in range(batch_size):
            results.append(self._do_step(c_start[0][b], c_end[0][b], 
                                        token_maps, params,
                                        verbose=verbose))
            results_guide.append(self.interp(c_start[1][b], c_end[1][b], 
                                             params["magnitude"], 
                                             params['lerp_threshold']))
        
        return (torch.stack(results),
                torch.stack(results_guide))

    def _do_step(self, c_start, c_end, token_maps, params, verbose=False):
        token_idxs = self._get_token_idxs(c_start, c_end, 
                                          token_maps,
                                          token_idxs=params['token_idxs'],
                                          token_range=params['token_range'],
                                          token_k=params['token_k'],
                                          token_largest=params['token_largest'],
                                          verbose=verbose)

        if token_idxs.shape[0] == 0:
            # no tokens selected, so no movement - still apply static multiplier
            result = c_start * params['static_mult']
        else:
            # interpolate - magnitude 1.0 means go all the way to end, 0.0 means don't move
            c_delta = self.interp(c_start, c_end, params['magnitude'], params['lerp_threshold'])

            # compute mask to restrict to subset and larger/smallest features to interpolate between
            mask = self._embed_topk_mask(c_delta, token_idxs,
                                        k=params['embed_k'], 
                                        embed_range=params['embed_range'],
                                        embed_idxs=params['embed_idxs'],
                                        largest=params['embed_largest'],
                                        verbose=verbose)
            mask = torch.from_numpy(mask).to(c_start.device)    
            # values we want to change 
            delta = c_delta * mask.double()
            delta_max = delta.max().item()
            delta_min = delta.min().item()
            # blur is a conv with special kernel that bleeds values only directly down
            if params['do_bleed']:
                delta = bleed(delta)
            # blur will compound values if they are already in a vertical line, this ensures the original max at least doesn't go up
            delta = torch.clip(delta, delta_min, delta_max)
            # values we want to stay the same
            static = c_start * torch.logical_not(mask).float()
            # combine together with elementwise addition since there is no overlap
            result = delta * params['delta_mult'] + \
                     static * params['static_mult']
            if verbose:
                plt.close()
                c_diff = c_start.sub(result)
                width_ratio = result.shape[1]/result.shape[0]
                fig, axs = plt.subplots(7, 1, figsize=(width_ratio * 4, 7 * 4))
                axs[0].imshow(c_start.cpu().numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                axs[0].axis('off')
                axs[0].set_title("Source Embedding - start point")
                axs[1].imshow(c_end.cpu().numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                axs[1].axis('off')
                axs[1].set_title("Target Embedding - end point")
                axs[2].imshow(mask.cpu().numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                axs[2].axis('off')
                axs[2].set_title("Binary mask - 77 tokens x 768 embedding dimensions")
                axs[3].imshow(delta.cpu().numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                axs[3].axis('off')
                axs[3].set_title("Masked Delta Embedding - interpolated embedding after masking")
                axs[4].imshow(static.cpu().numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                axs[4].axis('off')
                axs[4].set_title("Masked Static Embedding - original embedding after masking")
                axs[5].imshow(c_diff.cpu().numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                axs[5].axis('off')
                axs[5].set_title("Changes - start - result")
                axs[6].imshow(result.cpu().numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                axs[6].axis('off')
                axs[6].set_title("Result Embedding - delta + static (after scaling applied to each)")            
                fig.tight_layout()            
                plt.draw_all()

        return result 

    def _get_token_idxs(self, embed_start, embed_end, token_maps,
                        token_idxs=None,
                        token_range=None,
                        token_k=None,
                        token_largest=None,
                        verbose=False):
        T, E = embed_start.shape

        if token_k is not None and token_largest is not None:
            # force into 0-token_count range
            k = token_k = max(min(embed_start.shape[0], token_k), 0)
            if k == embed_start.shape[0]:
                # max number of tokens selected
                selected_idxs = np.array([i for i in range(k)])
            elif k == 0:
                # no tokens selected
                selected_idxs = np.array([])
            else:
                # compare token embeddings
                sim = F.cosine_similarity(embed_start.float(), embed_end.float()) 
                if token_largest:
                    token_repeat_mask = torch.Tensor(
                        [int(p1 != p2) for (p1,p2) 
                            in zip(token_maps[0], 
                                   token_maps[1])
                            ]).to(embed_start.device)
                    sim *= token_repeat_mask
                selected_idxs = torch.topk(sim.float(), 
                                        k=k, 
                                        dim=0,
                                        largest=token_largest)[1]
                selected_idxs = selected_idxs.cpu().numpy()
        elif token_range is not None:
            token_range = valid_range(T, r=token_range, idxs=token_idxs)
            selected_idxs = [i for i in range(token_range['start'], token_range['end'])] \
                            if token_idxs is None else token_idxs
            selected_idxs = np.array(selected_idxs)
        else:
            selected_idxs = [] if token_idxs is None else token_idxs
            selected_idxs = np.array(selected_idxs)

        if verbose:            
            print(f"token k: {token_k}\tselected max: {selected_idxs.max()} min: {selected_idxs.min()}") 

        return selected_idxs

    def _embed_topk_mask(self, embeddings, token_idxs,
                         k=None, 
                         embed_range=None, 
                         embed_idxs=None,
                         largest=True,
                         verbose=False):
        T, E = embeddings.shape
        
        embed_range = valid_range(E, r=embed_range, idxs=embed_idxs)
        
        total_embed_idxs = embed_idxs.shape[0] if embed_idxs is not None else \
                embed_range['end']-embed_range['start']
        k = k if k else total_embed_idxs
        # force into 0 - total embedding indexes range        
        k = max(min(k,total_embed_idxs), 0)        
        
        embed_idxs = embed_idxs if embed_idxs is not None else \
            [i for i in range(embed_range['start'], embed_range['end'])]
                        
        # select top/bottom k values from each token embedding specified in token_idxs
        embeddings_slice = embeddings[token_idxs, embed_range['start']:embed_range['end']]                
        selected_idxs = torch.topk(embeddings_slice.float(), 
                                   k=k, 
                                   dim=1, 
                                   largest=largest)[1].cpu().numpy()
        # shift indexes over by the range min, to account for slice index reset
        if embed_range['start'] > 0:
            selected_idxs = [[i2+embed_range['start'] for i2 in i] for i in selected_idxs]
            selected_idxs = np.array(selected_idxs)
        if verbose:            
            print(f"embed k: {k}\tembed range: {embed_range['start']} - {embed_range['end']}\tembedding slice shape: {embeddings_slice.shape}\tselected max: {selected_idxs.max()} min: {selected_idxs.min()}") 
        # create a lookup mapping between the 0-N index that top_idxs uses to actual token index
        token_idx_lookup = {n:i for (i,n) in enumerate(token_idxs)}
        # creates a TxR size array of bools with True only when 
        # t is in token_idxs and r is in the top_idx for that token
        mask = np.array([
            [(t in token_idxs) and 
             (r in selected_idxs[token_idx_lookup[t]]) and
             (r in embed_idxs)
                for r in range(E)
            ] for t in range(T)
        ])
        return mask  

class SampleConfigTransform(AbstractTransform):
    def __init__(self, target: dict, args: dict):
        super().__init__(args)
        self.target = target

    def to_json(self) -> Dict[str, Any]:
        json = super().to_json()
        json["target"] = self.target
        return json

    @classmethod
    def from_json(cls, json: dict, **kwargs):
        """
        Instantiates an object of this type from the given dictionary.
        The dictionary is expected to be the output from `to_json` and
        together allows for easy deserialization/serialization.

        Args:
            json (dict): JSON compatible dictionary containing all values needed
            to reinstantiate a serialized instance.

        Returns:
            `cls`: Deserialized instance
        """
        return cls(json["target"], json["args"]) 
        
    def apply(self, source, steps=1, verbose=False): 
        if len(self.param_lerp_keys) == 0 or \
            all([k not in self.args for k in self.param_lerp_keys]):
            steps = 1
            if verbose:
                print(f"No valid interpolatable parameters found. Set them as values for 'lerp_keys' in config dictionary")    
        for s in range(max(1,steps)):
            if verbose:
                print(f"[{s+1}/{steps}] {(s+1)/steps}%")
            params = self.lerp_params(self.args, (s+1)/steps, verbose=verbose)
            config = self.step(source, self.target, params, 
                                        verbose=verbose)
            self.step_results.append(config)
            yield config

    def step(self, source, target, params, verbose=False):
        raise NotImplementedError

class PromptSequenceTransform(AbstractTransform):
    def __init__(self, target: dict, args: dict):
        super().__init__(args)
        self.target = target
        self.parser = lark.Lark(r"""
        !start: (prompt | /[][():]/+)*
        prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
        !emphasized: "(" prompt ")"
                | "(" prompt ":" prompt ")"
                | "[" prompt "]"
        scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER "]"
        alternate: "[" prompt ("|" prompt)+ "]"
        WHITESPACE: /\s+/
        plain: /([^\\\[\]():|]|\\.)+/
        %import common.SIGNED_NUMBER -> NUMBER
        """)

    def to_json(self) -> Dict[str, Any]:
        json = super().to_json()
        json["target"] = self.target
        return json

    @classmethod
    def from_json(cls, json: dict, **kwargs):
        """
        Instantiates an object of this type from the given dictionary.
        The dictionary is expected to be the output from `to_json` and
        together allows for easy deserialization/serialization.

        Args:
            json (dict): JSON compatible dictionary containing all values needed
            to reinstantiate a serialized instance.

        Returns:
            `cls`: Deserialized instance
        """
        return cls(json["target"], json["args"]) 
        
    def apply(self, source, steps=1, verbose=False): 
        sequence = self.get_prompt_sequence  
        for s in range(max(1,steps)):
            if verbose:
                print(f"[{s+1}/{steps}] {(s+1)/steps}%")
            params = self.lerp_params(self.args, (s+1)/steps, verbose=verbose)
            config = self.step(source, self.target, params, 
                                        verbose=verbose)
            self.step_results.append(config)
            yield config

    def step(self, source, target, params, verbose=False):
        raise NotImplementedError

    def get_prompt_sequence(self, prompts, steps):
        """
        >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
        >>> g("test")
        [[10, 'test']]
        >>> g("a [b:3]")
        [[3, 'a '], [10, 'a b']]
        >>> g("a [b: 3]")
        [[3, 'a '], [10, 'a b']]
        >>> g("a [[[b]]:2]")
        [[2, 'a '], [10, 'a [[b]]']]
        >>> g("[(a:2):3]")
        [[3, ''], [10, '(a:2)']]
        >>> g("a [b : c : 1] d")
        [[1, 'a b  d'], [10, 'a  c  d']]
        >>> g("a[b:[c:d:2]:1]e")
        [[1, 'abe'], [2, 'ace'], [10, 'ade']]
        >>> g("a [unbalanced")
        [[10, 'a [unbalanced']]
        >>> g("a [b:.5] c")
        [[5, 'a  c'], [10, 'a b c']]
        >>> g("a [{b|d{:.5] c")  # not handling this right now
        [[5, 'a  c'], [10, 'a {b|d{ c']]
        >>> g("((a][:b:c [d:3]")
        [[3, '((a][:b:c '], [10, '((a][:b:c d']]
        """

        def collect_steps(steps, tree):
            l = [steps]
            class CollectSteps(lark.Visitor):
                def scheduled(self, tree):
                    tree.children[-1] = float(tree.children[-1])
                    if tree.children[-1] < 1:
                        tree.children[-1] *= steps
                    tree.children[-1] = min(steps, int(tree.children[-1]))
                    l.append(tree.children[-1])
                def alternate(self, tree):
                    l.extend(range(1, steps+1))
            CollectSteps().visit(tree)
            return sorted(set(l))

        def at_step(step, tree):
            class AtStep(lark.Transformer):
                def scheduled(self, args):
                    before, after, _, when = args
                    yield before or () if step <= when else after
                def alternate(self, args):
                    yield next(args[(step - 1)%len(args)])
                def start(self, args):
                    def flatten(x):
                        if type(x) == str:
                            yield x
                        else:
                            for gen in x:
                                yield from flatten(gen)
                    return ''.join(flatten(args))
                def plain(self, args):
                    yield args[0].value
                def __default__(self, data, children, meta):
                    for child in children:
                        yield from child
            return AtStep().transform(tree)

        def get_schedule(prompt):
            try:
                tree = self.parser.parse(prompt)
            except lark.exceptions.LarkError as e:
                if 0:
                    import traceback
                    traceback.print_exc()
                return [[steps, prompt]]
            return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

        promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
        return [promptdict[prompt] for prompt in prompts]    
    

class NullInversionTransform(AbstractTransform):
    def __init__(self, target: dict, args: dict):
        super().__init__(args)
        self.target = target
        # def __init__(self, unet, vae, tokenizer, text_encoder, scheduler=None, context=None, device="cuda"):
        self.scheduler = self.args.get("scheduler", None)
        self.unet = self.args.get("unet")
        self.vae = self.args.get("vae")
        self.tokenizer = self.args.get("tokenizer")
        self.text_encoder = self.args.get("text_encoder")
        self.context = self.args.get("context")
        self.device = self.args.get("device")
        self.scheduler = DDIMScheduler(beta_start=0.00085, 
                                       beta_end=0.012, 
                                       beta_schedule="scaled_linear", 
                                       clip_sample=False,
                                       set_alpha_to_one=False) if self.scheduler is None else self.scheduler
        self.prompt = None

    def to_json(self) -> Dict[str, Any]:
        json = super().to_json()
        json["target"] = self.target
        return json

    @classmethod
    def from_json(cls, json: dict, **kwargs):
        """
        Instantiates an object of this type from the given dictionary.
        The dictionary is expected to be the output from `to_json` and
        together allows for easy deserialization/serialization.

        Args:
            json (dict): JSON compatible dictionary containing all values needed
            to reinstantiate a serialized instance.

        Returns:
            `cls`: Deserialized instance
        """
        return cls(json["target"], json["args"]) 
        
    def apply(self, source, steps=1, verbose=False): 
        self.source = source
        self.context = torch.cat([self.source.embeddings[2], self.source.embeddings[0]])
        self.prompt = self.source.data

    def _calc_sample(self, model_output, sample, alpha_prod_t2, alpha_prod_t):
        beta_prod_t = 1 - alpha_prod_t
        original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        sample_direction = (1 - alpha_prod_t2) ** 0.5 * model_output
        new_sample = alpha_prod_t2 ** 0.5 * original_sample + sample_direction
        return new_sample
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        prev_sample = self._calc_sample(model_output, sample, alpha_prod_t_prev, alpha_prod_t)
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        next_sample = self._calc_sample(model_output, sample, alpha_prod_t_next, alpha_prod_t)
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.source.opt.scale
        noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
                latents = self.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents
        
    def ddim_inversion_only(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        image_gt = load_512(image_path, *offsets)
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        return (image_gt, image_rec), ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list
    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
    