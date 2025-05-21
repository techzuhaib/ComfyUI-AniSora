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

import .src.wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool


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
    
        if args.offload_model is None:
            args.offload_model = False if world_size > 1 else True
            logging.info(
                f"offload_model is not specified, set to {args.offload_model}.")
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size)
        else:
            assert not (
                args.t5_fsdp or args.dit_fsdp
            ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
            assert not (
                args.ulysses_size > 1 or args.ring_size > 1
            ), f"context parallel are not supported in non-distributed environments."
    
        if args.ulysses_size > 1 or args.ring_size > 1:
            assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
            from xfuser.core.distributed import (initialize_model_parallel,
                                                 init_distributed_environment)
            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size())
    
            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=args.ring_size,
                ulysses_degree=args.ulysses_size,
            )
    
        if args.use_prompt_extend:
            if args.prompt_extend_method == "dashscope":
                prompt_expander = DashScopePromptExpander(
                    model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
            elif args.prompt_extend_method == "local_qwen":
                prompt_expander = QwenPromptExpander(
                    model_name=args.prompt_extend_model,
                    is_vl="i2v" in args.task,
                    device=rank)
            else:
                raise NotImplementedError(
                    f"Unsupport prompt_extend_method: {args.prompt_extend_method}")
    
        cfg = WAN_CONFIGS[args.task]
        if args.ulysses_size > 1:
            assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."
    
        logging.info(f"Generation job args: {args}")
        logging.info(f"Generation model config: {cfg}")
    
        if dist.is_initialized():
            base_seed = [args.base_seed] if rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            args.base_seed = base_seed[0]
    
        if "t2v" in args.task or "t2i" in args.task:
            opt_dir=args.image
            with open(args.prompt,"r")as f:
                lines=f.read().strip("\n").split("\n")
            # if args.prompt is None:
            #     args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
            for idx,line in enumerate(lines):
                args.save_file="%s/%s.mp4"%(opt_dir,idx)
                prompt,image=line.split("@@")
                args.image=image
                args.prompt=prompt
                logging.info(f"Input prompt: {args.prompt}")
                if args.use_prompt_extend:
                    logging.info("Extending prompt ...")
                    if rank == 0:
                        prompt_output = prompt_expander(
                            args.prompt,
                            tar_lang=args.prompt_extend_target_lang,
                            seed=args.base_seed)
                        if prompt_output.status == False:
                            logging.info(
                                f"Extending prompt failed: {prompt_output.message}")
                            logging.info("Falling back to original prompt.")
                            input_prompt = args.prompt
                        else:
                            input_prompt = prompt_output.prompt
                        input_prompt = [input_prompt]
                    else:
                        input_prompt = [None]
                    if dist.is_initialized():
                        dist.broadcast_object_list(input_prompt, src=0)
                    args.prompt = input_prompt[0]
                    logging.info(f"Extended prompt: {args.prompt}")
    
                logging.info("Creating WanT2V pipeline.")
                wan_t2v = wan.WanT2V(
                    config=cfg,
                    checkpoint_dir=args.ckpt_dir,
                    device_id=device,
                    rank=rank,
                    t5_fsdp=args.t5_fsdp,
                    dit_fsdp=args.dit_fsdp,
                    use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
                    t5_cpu=args.t5_cpu,
                )
                logging.info(
                    f"Generating {'image' if 't2i' in args.task else 'video'} ...")
                video = wan_t2v.generate(
                    args.prompt,
                    size=SIZE_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model)
        else:
            if args.prompt is None:
                args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
            if args.image is None:
                args.image = EXAMPLE_PROMPT[args.task]["image"]
            logging.info(f"Input prompt: {args.prompt}")
            logging.info(f"Input image: {args.image}")
    
            opt_dir=args.image
            with open(args.prompt,"r",encoding="gbk")as f:
                lines=f.read().strip("\n").split("\n")
            logging.info("Creating WanI2V pipeline.")
            wan_i2v = wan.WanI2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
                t5_cpu=args.t5_cpu,
            )
    
            for idx,line in enumerate(lines):
                args.save_file="%s/%s.mp4"%(opt_dir,idx)
                prompt,image=line.split("@@")
                args.image=image
                args.prompt=prompt
                img = Image.open(args.image).convert("RGB")
                if args.use_prompt_extend:
                    logging.info("Extending prompt ...")
                    if rank == 0:
                        prompt_output = prompt_expander(
                            args.prompt,
                            tar_lang=args.prompt_extend_target_lang,
                            image=img,
                            seed=args.base_seed)
                        if prompt_output.status == False:
                            logging.info(
                                f"Extending prompt failed: {prompt_output.message}")
                            logging.info("Falling back to original prompt.")
                            input_prompt = args.prompt
                        else:
                            input_prompt = prompt_output.prompt
                        input_prompt = [input_prompt]
                    else:
                        input_prompt = [None]
                    if dist.is_initialized():
                        dist.broadcast_object_list(input_prompt, src=0)
                    args.prompt = input_prompt[0]
                    logging.info(f"Extended prompt: {args.prompt}")
                logging.info("Generating video ...")
                if os.path.exists(args.save_file)==False:
                    video = wan_i2v.generate(
                        args.prompt,
                        img,
                        max_area=MAX_AREA_CONFIGS[args.size],
                        frame_num=args.frame_num,
                        shift=args.sample_shift,
                        sample_solver=args.sample_solver,
                        sampling_steps=args.sample_steps,
                        guide_scale=args.sample_guide_scale,
                        seed=args.base_seed,
                        offload_model=args.offload_model)
        
        return (video,)

