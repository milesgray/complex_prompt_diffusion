
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

from base import AbstractPromptTransform
from cpd.util import from_json
from cpd.embeddings.prompts import ComplexPrompt

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
