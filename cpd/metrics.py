import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

def spherical_dist(x: Tensor, y: Tensor, reduce: bool=False) -> Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    dist = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
    if reduce:
        dist = dist.mean()
    return dist

def euclidean_dist(x: Tensor, y: Tensor, reduce: bool=False) -> Tensor:        
    dist = (x.pow(2) - y.pow(2)).sqrt()
    if reduce:
        dist = dist.mean()
    return dist

def cosine_sim(x: Tensor, y: Tensor):
    return F.cosine_similarity(x.float(), y.float()) 