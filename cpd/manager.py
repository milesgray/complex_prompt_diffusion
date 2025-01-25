
import traceback

import torch
from torch import autocast
import open_clip

from cpd.util import from_json, seed_everything, CudaMon, safe_to
from cpd.vram import setup_for_low_vram
from cpd.samplers import make as make_sampler
from cpd.samplers.extension import make as make_sampler_extension

class DiffusionModelManager:
    def __init__(self, checkpoint_file, model_key='model', verbose=False):
        try:
            self.clog = CudaMon("manager", verbose=verbose)
            self.clog("__init__","enter")
            self.model_dict = torch.load(checkpoint_file, map_location="cpu")
            self.model_dict["decode"] = self.model_dict[model_key].first_stage_model.decode
            self.model_dict["vae"] = self.model_dict[model_key].first_stage_model
            self.model_dict["unet"] = self.model_dict[model_key].model.diffusion_model
            self.model_dict["embedder"] = self.model_dict[model_key].cond_stage_model
            self.model_dict["tokenizer"] = open_clip.tokenize

            self.model_dict[model_key] = self.model_dict[model_key].half()
            self.model_dict[model_key].model.diffusion_model.dtype = torch.float16
            self.model = self.model_dict[model_key]
            self.clog("__init__","model loaded")            
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                
            for k,v in self.model_dict.items():
                self.model_dict[k] = safe_to(v, dtype=torch.float16)
                if hasattr(self.model_dict[k], "dtype"):
                    try:
                        if self.model_dict[k].dtype != torch.float16:
                            self.model_dict[k].dtype = torch.float16
                    except:
                        pass
            #self.model = self.model.to(device)
            self.model_dict["embedder"].device = "cuda"
            setup_for_low_vram(self.model, use_medvram=True)
            self.clog("__init__","set in vram")            
            self.model.eval()
            self.z_channels = self.model. \
                                first_stage_model. \
                                    encoder. \
                                        conv_out.weight.shape[0] // 2                                        
        except KeyError as ke:
            raise KeyError(f"Checkpoint file has no '{model_key}' key, change checkpoint file or key")        
        print(f"[manager.__init__]\t[exit]\t-\t{torch.cuda.memory_allocated()}")

    def process_txt2img(self, config: dict):
        sampler = self._make_sampler(config)
        self.clog("process_txt2img","sampler made")
        cpe = self._make_embedding(sampler, config)
        cpe.sampler = sampler
        self.clog("process_txt2img","embedding made")

        render_args = config.get("render", {})
        render_args["score_corrector"] = self._make_score_corrector(render_args)
        self.clog("process_txt2img","render args setup, rendering")
        outs = cpe.render(verbose=False, **render_args)
        img = outs[0]

        self.clog("process_txt2img","text render sampled")
        return img

    def process_img2img(self, img, mask, prompt, config: dict): 
        sampler = self._make_sampler(config)
        self.clog("process_txt2img","sampler made")
        x = self._render_img(sampler, img, mask, prompt,
                             batch_size=config.get("batch_size", 1),
                             seed=config.get("seed", 42),
                             reset_seed=config.get("reset_seed", True))
        self.clog("process_txt2img","image render sampled")
        img = self._create_image(x)
        self.clog("process_txt2img","image created")
        return img

    def _make_embedding(self, sampler, config: dict):
        json = config.get("prompt_json")
        return from_json(json, model=self.model_dict, sampler=sampler, logger=print)

    def _make_score_corrector(self, render_args: dict):
        name = render_args.get("score_corrector", None)
        x_threshold = render_args.get("score_corrector_x_threshold", None)
        e_threshold = render_args.get("score_corrector_e_threshold", None)
        if name is not None:
            score_corrector = make_sampler_extension({"name": name, "args": {"threshold_x": x_threshold, "threshold_e": e_threshold}})
        else:
            score_corrector = None
        return score_corrector

    def _make_sampler(self, config: dict):
        sampler_config = config.get("sampler", {"name": "DDIM", "args": {}})        
        
        assert "args" in sampler_config and "name" in sampler_config
        
        return make_sampler(sampler_config, args={"model": self.model_dict})

    def _get_unconditional_embeddings(self, batch_size: int=1):
        with torch.no_grad():
            return self.model.get_learned_conditioning(batch_size * [""])
    
    def _get_conditioning_embeddings(self, prompt: list, batch_size=1):                
        with torch.no_grad():
            return self.model.get_learned_conditioning(batch_size * prompt)      

    def _create_image(self, x):
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        img = x.cpu().squeeze().permute(1, 2, 0)
        img = img.mul(255).to(torch.uint8).numpy()
        return img
    
    def _render_text(self, sampler, prompt,
                     batch_size=1,
                     start_code=None,
                     seed=-1,
                     reset_seed=False):
        uc = self._get_unconditional_embeddings(batch_size=batch_size)
        c = self._get_conditioning_embeddings(prompt, batch_size=batch_size)
        assert c.shape == uc.shape
        self.clog("_render_text","emb made")
        
        if reset_seed: seed_everything(seed)
        
        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            samples_x = sampler.sample(conditioning=c,
                                       unconditional_conditioning=uc,
                                       start_code=start_code)
            self.clog("_render_text","sample made")
            
            x = self.model.decode_first_stage(samples_x)
            self.clog("_render_text","sample decoded")
            return x    

    def _render_img(self, sampler, img, mask, prompt,
                    batch_size=1,
                    seed=-1,
                    reset_seed=False):
        uc = self._get_unconditional_embeddings(batch_size=batch_size)
        c = self._get_conditioning_embeddings(prompt, batch_size=batch_size)
        assert c.shape == uc.shape
        self.clog("_render_img","emb made")
        if reset_seed: seed_everything(seed)
        
        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            x = sampler.sample_img(img, mask, 
                                    conditioning=c, 
                                    unconditional_conditioning=uc)
            self.clog("_render_img","sample made")                                    
            return x