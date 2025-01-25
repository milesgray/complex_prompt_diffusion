import numpy as np
import torch
from einops import rearrange

from cpd.samplers.extension.registry import register

@register("none")
class ScoreCorrector:
    def __init__(self, threshold_x=None, threshold_e=None):
        self.threshold_x = threshold_x
        self.threshold_e = threshold_e

    def apply(self, x, t, **kwargs):
        x = self._apply(x, **kwargs)
        return x

    def modify_score(self, e_t, x, t, c, **kwargs):
        t = t.item() if isinstance(t, torch.Tensor) else t
        kwargs["t"] = t
        if self.threshold_x:
            kwargs["name"] = "x"
            kwargs["threshold"] = self.threshold_x
            x = self._apply(x, **kwargs)
        if self.threshold_e:
            kwargs["name"] = "e_t"
            kwargs["threshold"] = self.threshold_e
            e_t = self._apply(e_t, **kwargs)
        verbose = kwargs.get("verbose", False)
        if verbose:
            print(f"[{int(t)}] => e(t): {e_t.min():0.5f} {e_t.max():0.5f}, {e_t.mean():0.5f}")
        return e_t

    def __call__(self, x, **kwargs):
        if isinstance(x, dict):
            if "t" not in x and "sigma" in x:
                x["t"] = x["sigma"]
            return self._apply(**x)
        else:
            return self._apply(x, **kwargs)

    def _apply(self, x, **kwargs):
        t = kwargs.get("t", 0)
        name = kwargs.get("name","")
        verbose = kwargs.get("verbose", False)        
        return x

@register("static_thresholding")
class StaticThresholding(ScoreCorrector):
    def __init__(self, threshold_x=None, threshold_e=None):
        self.threshold_x = threshold_x
        self.threshold_e = threshold_e

    def _apply(self, x, **kwargs):
        t = kwargs.get("t", 0)
        verbose = kwargs.get("verbose", False)
        threshold = kwargs.get("threshold", 1)
        name = kwargs.get("name", "x")
        x = x.float()
       
        torch.clamp_(x, -1*threshold, threshold)
               
        return x.half()
@register("dynamic_thresholding")
class DynamicThresholding(ScoreCorrector):
    def __init__(self, threshold_x=None, threshold_e=None):
        self.threshold_x = threshold_x
        self.threshold_e = threshold_e

    def _apply(self, x, **kwargs):
        t = kwargs.get("t", 0)
        verbose = kwargs.get("verbose", False)
        threshold = kwargs.get("threshold", 99.66)
        name = kwargs.get("name", "x")
        x = x.float()
        
        s = np.percentile(
            np.abs(x.cpu()), 
            threshold, 
            axis=tuple(range(1,x.ndim))
        )
       
        s = np.max(np.append(s,1.0))
        torch.clamp_(x, -1*s, s)

        return x.half()

@register("dynanormic_thresholding")
class DynanormicThresholding(ScoreCorrector):
    def __init__(self, threshold_x=None, threshold_e=None):
        self.threshold_x = threshold_x
        self.threshold_e = threshold_e

    def _apply(self, x, **kwargs):
        t = kwargs.get("t", 0)
        verbose = kwargs.get("verbose", False)
        threshold = kwargs.get("threshold", 99.66)
        name = kwargs.get("name", "x")

        if threshold > 1 and threshold <= 100:
            threshold = threshold / 100
        
        x = x.float()

        s = torch.quantile(
            torch.abs(x).reshape((x.shape[0], -1)), 
            threshold, 
            dim=1
        )
        s = torch.maximum(
            s,
            1 * torch.ones_like(s).to(s.device),
        )[(...,) + (None,) * (x.ndim - 1)]
        x = torch.clamp(x, -s, s) 
        x = x / s
        
        return x.half()

@register("scaled_dynamic_perc_thresholding")
class ScaledDynamicPercThresholding(ScoreCorrector):
    def __init__(self, threshold_x=None, threshold_e=None):
        self.threshold_x = threshold_x
        self.threshold_e = threshold_e

    def _apply(self, x, **kwargs):
        t = kwargs.get("t", 0)
        verbose = kwargs.get("verbose", False)
        threshold = kwargs.get("threshold", 99.66)
        name = kwargs.get("name", "x")
        
        x = x.float()

        x_max, x_min = x.max(), x.min()
        x = (x-x_min)/(x_max-x_min)
        x = 2 * x - 1.
        
        s = np.percentile(
            np.abs(x.cpu()), 
            threshold, 
            axis=tuple(range(1,x.ndim))
        )
        s = np.max(np.append(s,1.0))
        torch.clamp_(x, -1*s, s)

        x = (x + 1) / 2
        x = (x_max-x_min) * x + x_min
        return x.half()

@register("renorm_thresholding")
class RenormThresholding(ScoreCorrector):
    def __init__(self, threshold_x=None, threshold_e=None):
        self.threshold_x = threshold_x
        self.threshold_e = threshold_e

    def _apply(self, x, **kwargs):
        t = kwargs.get("t", 0)
        verbose = kwargs.get("verbose", False)
        threshold = kwargs.get("threshold", 99.66)
        name = kwargs.get("name", "x")
       
        x = x.float()
        x_max, x_min = x.max(), x.min()
        
        x = (x-x_min)/(x_max-x_min)
        x = 2 * x - 1.
        
        if threshold > 1 and threshold <= 100:
            threshold = threshold / 100

        s = torch.quantile(
            rearrange(x.float(), 'b ... -> b (...)').abs(), 
            threshold, 
            dim=-1
        )
                   
        s.clamp_(min=1.0)
        torch.clamp_(x, -1*s, s)
        
        x = (x + 1) / 2
        x = (x_max-x_min) * x + x_min
        return x.half()

@register("norm_thresholding")
class NormThresholding(ScoreCorrector):
    def __init__(self, threshold_x=None, threshold_e=None):
        self.threshold_x = threshold_x
        self.threshold_e = threshold_e

    def _apply(self, x, **kwargs):
        t = kwargs.get("t", 0)
        verbose = kwargs.get("verbose", False)
        threshold = kwargs.get("threshold", 99.66)
        name = kwargs.get("name", "x")
        x = x.float()
        threshold = threshold / 100
        threshold = threshold * x_max
        
        s = x.pow(2) \
            .flatten(1) \
            .mean(1) \
            .sqrt() \
            .clamp(min=threshold)
            
        x = x * (threshold / s)

        return x.half()

@register("scaled_norm_thresholding")
class ScaledNormThresholding(ScoreCorrector):
    def __init__(self, threshold_x=None, threshold_e=None):
        self.threshold_x = threshold_x
        self.threshold_e = threshold_e

    def _apply(self, x, **kwargs):
        t = kwargs.get("t", 0)
        verbose = kwargs.get("verbose", False)
        threshold = kwargs.get("threshold", 99.66)
        name = kwargs.get("name", "x")
        x = x.float()
        x_max, x_min = x.max(), x.min()
        
        x = (x-x_min)/(x_max-x_min)        
        x = 2 * x - 1.

        threshold = threshold / 100
        threshold = threshold * x_max
        
        s = x.pow(2) \
            .flatten(1) \
            .mean(1) \
            .sqrt() \
            .clamp(min=threshold)
        x = x * (threshold / s)

        x = (x + 1) / 2
        x = (x_max-x_min) * x + x_min

        return x.half()

@register("spatial_norm_thresholding")
class SpatialNormThresholding(ScoreCorrector):
    def __init__(self, threshold_x=None, threshold_e=None):
        self.threshold_x = threshold_x
        self.threshold_e = threshold_e

    def _apply(self, x, **kwargs):
        t = kwargs.get("t", 0)
        verbose = kwargs.get("verbose", False)
        threshold = kwargs.get("threshold", 99.66)
        name = kwargs.get("name", "x")
        x = x.float()
        s = x.pow(2).mean(1, keepdim=True).sqrt().clamp(min=threshold)
        x = x * (threshold / s)

        return x.half()

@register("scaled_spatial_norm_thresholding")
class ScaledSpatialNormThresholding(ScoreCorrector):
    def __init__(self, threshold_x=None, threshold_e=None):
        self.threshold_x = threshold_x
        self.threshold_e = threshold_e

    def _apply(self, x, **kwargs):
        t = kwargs.get("t", 0)
        verbose = kwargs.get("verbose", False)
        threshold = kwargs.get("threshold", 99.66)
        name = kwargs.get("name", "x")
        x = x.float()
        x_max, x_min = x.max(), x.min()
        
        x = (x-x_min)/(x_max-x_min)
        x = 2 * x - 1.
        
        threshold = threshold / 100
        threshold = threshold * x_max

        s = x.pow(2) \
            .mean(1, keepdim=True) \
            .sqrt() \
            .clamp(min=threshold)
            
        x = x * (threshold / s)

        x = (x + 1) / 2
        x = (x_max-x_min) * x + x_min
        
        return x.half()