import warnings
import importlib.util
from inspect import isfunction
from typing import Optional
from collections import OrderedDict
from dataclasses import fields
from typing import Any, Tuple
from collections import defaultdict
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
_USE_MEMORY_EFFICIENT_ATTENTION = 0
if _USE_MEMORY_EFFICIENT_ATTENTION:
    import xformers
    import xformers.ops


#from configuration_utils import ConfigMixin, register_to_config
#from modeling_utils import ModelMixin
#from embeddings import ImagePositionalEmbeddings
  
_is_xformers_available = importlib.util.find_spec("xformers") is not None    
def is_xformers_available():
    return _is_xformers_available
from cpd.models.util import checkpoint

heat_maps = defaultdict(list)
all_heat_maps = []

def clear_heat_maps():
    global heat_maps, all_heat_maps
    heat_maps = defaultdict(list)
    all_heat_maps = []

def next_heat_map():
    global heat_maps, all_heat_maps
    all_heat_maps.append(heat_maps)
    heat_maps = defaultdict(list)

def get_global_heat_map(last_n: int = None, idx: int = None, factors=None):
    global heat_maps, all_heat_maps

    if idx is not None:
        heat_maps2 = [all_heat_maps[idx]]
    else:
        heat_maps2 = all_heat_maps[-last_n:] if last_n is not None else all_heat_maps

    if factors is None:
        factors = {1, 2, 4, 8, 16, 32}

    all_merges = []

    for heat_map_map in heat_maps2:
        merge_list = []

        for k, v in heat_map_map.items():
            if k in factors:
                merge_list.append(torch.stack(v, 0).mean(0))

        all_merges.append(merge_list)

    maps = torch.stack([torch.stack(x, 0) for x in all_merges], dim=0)
    return maps.sum(0).cuda().sum(2).sum(0)



def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class CrossAttention(nn.Module):
    hypernetwork = None
    noise_cond = False
    def set_hypernetwork(hypernetwork):
        CrossAttention.hypernetwork = hypernetwork
    def set_noise_cond(nc):
        CrossAttention.noise_cond = nc

    # mix conditioning vectors for prompts
    def prompt_mixing(prompt_body, batch_size):
        if "|" in prompt_body:
            prompt_parts = prompt_body.split("|")
            prompt_total_power = 0
            prompt_sum = None
            for prompt_part in prompt_parts:
                prompt_power = 1
                if ":" in prompt_part:
                    prompt_sub_parts = prompt_part.split(":")
                    try:
                        prompt_power = float(prompt_sub_parts[1])
                        prompt_part = prompt_sub_parts[0]
                    except:
                        print("Error parsing prompt power! Assuming 1")
                prompt_vector = CrossAttention._hack_model.get_learned_conditioning([prompt_part])
                if prompt_sum is None:
                    prompt_sum = prompt_vector * prompt_power
                else:
                    prompt_sum = prompt_sum + (prompt_vector * prompt_power)
                prompt_total_power = prompt_total_power + prompt_power
            return CrossAttention.fix_batch(prompt_sum / prompt_total_power, batch_size)
        else:
            return CrossAttention.fix_batch(CrossAttention._hack_model.get_learned_conditioning([prompt_body]), batch_size)

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self._slice_size = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def daam_forward(self, x, context=None, mask=None):
        batch_size, sequence_length, dim = x.shape

        use_context = context is not None

        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.reshape_heads_to_batch_dim(q)
        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of
        hidden_states = self._attention(q, k, v, sequence_length, dim, use_context=use_context)

        return self.to_out(hidden_states)
    
    @torch.no_grad()
    def _up_sample_attn(self, x, factor, method: str = 'bicubic'):
        weight = torch.full((factor, factor), 1 / factor**2, device=x.device)
        weight = weight.view(1, 1, factor, factor)

        h = w = int(math.sqrt(x.size(1)))
        maps = []
        x = x.permute(2, 0, 1)

        with torch.cuda.amp.autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.unsqueeze(1).view(map_.size(0), 1, h, w)
                if method == 'bicubic':
                    map_ = F.interpolate(map_, size=(55, 55), mode="bicubic", align_corners=False)
                    maps.append(map_.squeeze(1))
                else:
                    maps.append(F.conv_transpose2d(map_, weight, stride=factor).squeeze(1).cpu())

        maps = torch.stack(maps, 0).cpu()
        return maps

    def _attention(self, query, key, value, sequence_length, dim, use_context: bool = True):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx]) * self.scale
            )
            factor = int(math.sqrt(4096 // attn_slice.shape[1]))
            attn_slice = attn_slice.softmax(-1)

            if use_context:
                if factor >= 1:
                    factor //= 1
                    maps = self._up_sample_attn(attn_slice, factor)
                    global heat_maps
                    heat_maps[factor].append(maps)
                # print(attn_slice.size(), query.size(), key.size(), value.size())

            attn_slice = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states


    def forward(self, x, context=None, mask=None):
        h = self.heads

        q_in = self.to_q(x)
        context = default(context, x)
        if CrossAttention.hypernetwork is not None and context.shape[2] in CrossAttention.hypernetwork:
            if context.shape[1] == 77 and CrossAttention.noise_cond:
                context = context + (torch.randn_like(context) * 0.1)
            h_k, h_v = CrossAttention.hypernetwork[context.shape[2]]
            k_in = self.to_k(h_k(context))
            v_in = self.to_v(h_v(context))
        else:
            k_in = self.to_k(context)
            v_in = self.to_v(context)
        del context, x

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)

        stats = torch.cuda.memory_stats(q.device)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch

        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier
        steps = 1

        if mem_required > mem_free_total:
            steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))
            # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
            #      f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                               f'Need: {mem_required/64/gb:0.1f}GB free, Have:{mem_free_total/gb:0.1f}GB free')

        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k) * self.scale
            if exists(mask):
                mask1 = rearrange(mask, 'b ... -> b (...)')
                
                max_neg_value = -torch.finfo(s1.dtype).max
                mask2 = repeat(mask1, 'b j -> (b h) () j', h=h)
                del mask1
                s1.masked_fill_(~mask2, max_neg_value)
                del mask2

            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            del s1

            r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
            del s2

        del q, k, v

        r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
        del r1

        return self.to_out(r2)


# https://www.photoroom.com/tech/stable-diffusion-100-percent-faster-with-memory-efficient-attention/
class MemoryEfficientCrossAttention(nn.Module):
    hypernetwork = None
    noise_cond = False
    def set_hypernetwork(hypernetwork):
        MemoryEfficientCrossAttention.hypernetwork = hypernetwork
    def set_noise_cond(nc):
        MemoryEfficientCrossAttention.noise_cond = nc

    # mix conditioning vectors for prompts
    def prompt_mixing(prompt_body, batch_size):
        if "|" in prompt_body:
            prompt_parts = prompt_body.split("|")
            prompt_total_power = 0
            prompt_sum = None
            for prompt_part in prompt_parts:
                prompt_power = 1
                if ":" in prompt_part:
                    prompt_sub_parts = prompt_part.split(":")
                    try:
                        prompt_power = float(prompt_sub_parts[1])
                        prompt_part = prompt_sub_parts[0]
                    except:
                        print("Error parsing prompt power! Assuming 1")
                prompt_vector = CrossAttention._hack_model.get_learned_conditioning([prompt_part])
                if prompt_sum is None:
                    prompt_sum = prompt_vector * prompt_power
                else:
                    prompt_sum = prompt_sum + (prompt_vector * prompt_power)
                prompt_total_power = prompt_total_power + prompt_power
            return CrossAttention.fix_batch(prompt_sum / prompt_total_power, batch_size)
        else:
            return CrossAttention.fix_batch(CrossAttention._hack_model.get_learned_conditioning([prompt_body]), batch_size)

    
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q_in = self.to_q(x)
        context = default(context, x)
        if MemoryEfficientCrossAttention.hypernetwork is not None and context.shape[2] in MemoryEfficientCrossAttention.hypernetwork:
            if context.shape[1] == 77 and MemoryEfficientCrossAttention.noise_cond:
                context = context + (torch.randn_like(context) * 0.1)
            h_k, h_v = MemoryEfficientCrossAttention.hypernetwork[context.shape[2]]
            k_in = self.to_k(h_k(context))
            v_in = self.to_v(h_v(context))
        else:
            k_in = self.to_k(context)
            v_in = self.to_v(context)
        del context, x        

        b, _, _ = q_in.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q_in, k_in, v_in),
        )
        del q_in, k_in, v_in

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
        del q, k, v

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.
    Parameters:
        dim (:obj:`int`): The number of channels in the input and output.
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
        context_dim (:obj:`int`, *optional*): The size of the context vector for cross attention.
        gated_ff (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use a gated feed-forward network.
        checkpoint (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use checkpointing.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout=0.0,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        checkpoint: bool = True,
        disable_self_attn: bool = False,
    ):
        super().__init__()
        AttentionBuilder = MemoryEfficientCrossAttention if _USE_MEMORY_EFFICIENT_ATTENTION else CrossAttention
        self.attn1 = AttentionBuilder(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = AttentionBuilder(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def _set_attention_slice(self, slice_size):
        self.attn1._slice_size = slice_size
        self.attn2._slice_size = slice_size

    def forward(self, hidden_states, context=None):
        hidden_states = hidden_states.contiguous() if hidden_states.device.type == "mps" else hidden_states
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), context=context) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                   disable_self_attn=disable_self_attn)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in

class HyperLogic(torch.nn.Module):
    logic_multiplier = 1.0
    def __init__(self, dim, heads=0):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim*2)
        self.linear2 = torch.nn.Linear(dim*2, dim)

    def forward(self, _x):
        return _x + (self.linear2(self.linear1(_x)) * HyperLogic.logic_multiplier)

    
class BaseOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `BaseOutput` directly. Use the [`~utils.BaseOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and isinstance(first_field, dict):
            for key, value in first_field.items():
                self[key] = value
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())

