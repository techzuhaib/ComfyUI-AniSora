import os
os.environ["HF_ENDPOINT"]          = "https://hf-mirror.com"
import argparse
from datetime import datetime
import logging
import sys
import warnings

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import .src.anisoraV2_gpu.wan
from .src.anisoraV2_gpu.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from .src.anisoraV2_gpu.wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from .src.anisoraV2_gpu.wan.utils.utils import cache_video, cache_image, str2bool


class LoadAniSoraModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "./Wan2.1-I2V-14B-480P"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "AniSora"

    def load_model(self, model_path):
        model = model_path
        return (model,)


class Prompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "input_text"
    CATEGORY = "AniSora"

    def input_text(self, text):
        prompt = text
        return (prompt,)


class AniSora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "task": ("STRING", {"default": "t2v-14B"}),
                "ckpt_dir": ("MODEL",),
                "prompt": ("PROMPT",),
                "size": ("STRING", {"default": "1280*720"}),
                "frame_num": ("INT", {"default": 81}),
                "ulysses_size": ("INT", {"default": 4}),
                "ring_size": ("INT", {"default": 1}),
                "base_seed": ("INT", {"default": "42"}),
                "sample_solver": ("STRING", {"default": ['unipc', 'dpm++']}),
                "sample_steps": ("INT", {"default": 50}),
                "sample_shift": ("FLOAT", {"default": 5}),
                "sample_guide_scale": ("FLOAT", {"default": 5.0}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "AniSora"

    def generate(self, task, size, frame_num, ckpt_dir, ulysses_size, ring_size, prompt, 
                base_seed, sample_solver, sample_steps, sample_shift, sample_guide_scale):

        offload_model = None
        t5_fsdp = False
        t5_cpu = False
        dit_fsdp = False
                  
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank
        _init_logging(rank)
    
        if offload_model is None:
            offload_model = False if world_size > 1 else True
            
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size)
        else:
            assert not (
                t5_fsdp or dit_fsdp
            ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
            assert not (
                ulysses_size > 1 or ring_size > 1
            ), f"context parallel are not supported in non-distributed environments."
    
        if ulysses_size > 1 or ring_size > 1:
            assert ulysses_size * ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
            from xfuser.core.distributed import (initialize_model_parallel,
                                                 init_distributed_environment)
            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size())
    
            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=ring_size,
                ulysses_degree=ulysses_size,
            )
    
        if use_prompt_extend:
            if prompt_extend_method == "dashscope":
                prompt_expander = DashScopePromptExpander(
                    model_name=prompt_extend_model, is_vl="i2v" in task)
            elif prompt_extend_method == "local_qwen":
                prompt_expander = QwenPromptExpander(
                    model_name=prompt_extend_model,
                    is_vl="i2v" in task,
                    device=rank)
            else:
                raise NotImplementedError(
                    f"Unsupport prompt_extend_method: {prompt_extend_method}")
    
        cfg = WAN_CONFIGS[task]
        if ulysses_size > 1:
            assert cfg.num_heads % ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."
    
        if dist.is_initialized():
            base_seed = [base_seed] if rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            base_seed = base_seed[0]
    
        if "t2v" in task or "t2i" in task:
            opt_dir=image
            with open(prompt,"r")as f:
                lines=f.read().strip("\n").split("\n")
            for idx,line in enumerate(lines):
                save_file="%s/%s.mp4"%(opt_dir,idx)
                prompt,image=line.split("@@")
                image=image
                prompt=prompt

                if use_prompt_extend:
                    if rank == 0:
                        prompt_output = prompt_expander(
                            prompt,
                            tar_lang=prompt_extend_target_lang,
                            seed=base_seed)
                        if prompt_output.status == False:
                            input_prompt = prompt
                        else:
                            input_prompt = prompt_output.prompt
                        input_prompt = [input_prompt]
                    else:
                        input_prompt = [None]
                    if dist.is_initialized():
                        dist.broadcast_object_list(input_prompt, src=0)
                    prompt = input_prompt[0]
    
                wan_t2v = wan.WanT2V(
                    config=cfg,
                    checkpoint_dir=ckpt_dir,
                    device_id=device,
                    rank=rank,
                    t5_fsdp=t5_fsdp,
                    dit_fsdp=dit_fsdp,
                    use_usp=(ulysses_size > 1 or ring_size > 1),
                    t5_cpu=t5_cpu,
                )

                video = wan_t2v.generate(
                    prompt,
                    size=SIZE_CONFIGS[size],
                    frame_num=frame_num,
                    shift=sample_shift,
                    sample_solver=sample_solver,
                    sampling_steps=sample_steps,
                    guide_scale=sample_guide_scale,
                    seed=base_seed,
                    offload_model=offload_model)
        else:
            if prompt is None:
                prompt = EXAMPLE_PROMPT[task]["prompt"]
            if image is None:
                image = EXAMPLE_PROMPT[task]["image"]
    
            opt_dir=image
            with open(args.prompt,"r",encoding="gbk")as f:
                lines=f.read().strip("\n").split("\n")

            wan_i2v = wan.WanI2V(
                config=cfg,
                checkpoint_dir=.ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=t5_fsdp,
                dit_fsdp=.dit_fsdp,
                use_usp=(ulysses_size > 1 or ring_size > 1),
                t5_cpu=t5_cpu,
            )
    
            for idx,line in enumerate(lines):
                prompt,image=line.split("@@")
                image=image
                prompt=prompt
                img = Image.open(args.image).convert("RGB")
                
                if os.path.exists(save_file)==False:
                    video = wan_i2v.generate(
                        prompt,
                        img,
                        max_area=MAX_AREA_CONFIGS[size],
                        frame_num=frame_num,
                        shift=sample_shift,
                        sample_solver=sample_solver,
                        sampling_steps=sample_steps,
                        guide_scale=sample_guide_scale,
                        seed=base_seed,
                        offload_model=offload_model)
        
        return (video,)

