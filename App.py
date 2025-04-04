import os
import sys
import subprocess
import random
import argparse
import time
import tempfile
import json
import gc
import re
import shutil

import psutil
DEFAULT_CLEAR_CACHE = True if psutil.virtual_memory().total < 31 * 1024**3 else False

import torch
import gradio as gr
from PIL import Image, ImageOps
import cv2

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download

# Global variables
loaded_pipeline = None
loaded_pipeline_config = {}
cancel_flag = False
cancel_batch_flag = False
prompt_expander = None
last_selected_aspect_ratio = None

# ------------------------- Utility Functions -------------------------

def extract_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    return None

def generate_prompt_info(parameters):
    """
    Generate prompt info text from parameters dict.
    """
    details = ""
    details += f"Prompt: {parameters['prompt']}\n"
    details += f"Negative Prompt: {parameters['negative_prompt']}\n"
    
    # Add clarity for extension models
    if 'extension_segment' in parameters and parameters['extension_segment'] > 0:
        details += f"Used Model: {parameters['model_choice']} (Extension Model)\n"
    else:
        details += f"Used Model: {parameters['model_choice']}\n"
    
    if 'extension_model' in parameters:
        details += f"Extension Model: {parameters['extension_model']}\n"
    
    details += f"Number of Inference Steps: {parameters['inference_steps']}\n"
    details += f"CFG Scale: {parameters['cfg_scale']}\n"
    details += f"Sigma Shift: {parameters['sigma_shift']}\n"
    details += f"Seed: {parameters['seed']}\n"
    details += f"Number of Frames: {parameters['num_frames']}\n"
    
    if 'extend_factor' in parameters:
        details += f"Extend Factor: {parameters['extend_factor']}x\n"
    
    if 'num_segments' in parameters:
        details += f"Number of Segments: {parameters['num_segments']}\n"
    
    if 'extension_segment' in parameters:
        details += f"Extension Segment: {parameters['extension_segment']} of {parameters['total_extensions']}\n"
    
    if 'source_frame' in parameters:
        details += f"Used Last Frame From: {parameters['source_frame']}\n"
    
    if 'input_file' in parameters:
        if parameters.get('is_video', False):
            details += f"Input Video: {parameters['input_file']}\n"
        else:
            details += f"Input Image: {parameters['input_file']}\n"
    
    if 'denoising_strength' in parameters:
        if parameters.get('is_text_to_video', False) and not parameters.get('has_input_video', False):
            details += "Denoising Strength: N/A\n"
        else:
            details += f"Denoising Strength: {parameters['denoising_strength']}\n"
    
    if 'pr_rife_enabled' in parameters and parameters['pr_rife_enabled']:
        details += f"Practical-RIFE: Enabled, Multiplier: {parameters.get('pr_rife_multiplier', '(unspecified)')}\n"
    
    if 'segment_details' in parameters:
        if isinstance(parameters['segment_details'], list):
            for detail in parameters['segment_details']:
                if isinstance(detail, tuple):
                    i, text = detail
                    details += f"Extension segment {i}: {text}\n"
                else:
                    details += f"{detail}\n"
    
    if 'lora_details' in parameters:
        if parameters['lora_details']:
            details += f"LoRA Models: {parameters['lora_details']}\n"
        else:
            details += "LoRA Model: None\n"
    
    details += f"TeaCache Enabled: {parameters['enable_teacache']}\n"
    if parameters['enable_teacache']:
        details += f"TeaCache L1 Threshold: {parameters['tea_cache_l1_thresh']}\n"
        details += f"TeaCache Model ID: {parameters['tea_cache_model_id']}\n"
    
    details += f"Precision: {'FP8' if parameters['torch_dtype'] == 'torch.float8_e4m3fn' else 'BF16'}\n"
    details += f"Auto Crop: {'Enabled' if parameters.get('auto_crop', False) else 'Disabled'}\n"
    details += f"Final Resolution: {parameters['width']}x{parameters['height']}\n"
    
    if 'video_generation_duration' in parameters:
        details += f"Video Generation Duration: {parameters['video_generation_duration']:.2f} seconds"
        if parameters.get('include_minutes', False):
            details += f" / {parameters['video_generation_duration']/60:.2f} minutes"
        details += "\n"
    
    if 'generation_duration' in parameters:
        details += f"Total Processing Duration: {parameters['generation_duration']:.2f} seconds"
        if parameters.get('include_minutes', False):
            details += f" / {parameters['generation_duration']/60:.2f} minutes"
        details += "\n"
    
    return details

def merge_videos(video_files, output_dir="outputs"):
    filelist_path = os.path.join(tempfile.gettempdir(), "filelist.txt")
    with open(filelist_path, "w", encoding="utf-8") as f:
        for vf in video_files:
            f.write(f"file '{os.path.abspath(vf)}'\n")
    merged_video = get_next_filename(".mp4", output_dir=output_dir)
    cmd = f'ffmpeg -f concat -safe 0 -i "{filelist_path}" -c copy "{merged_video}"'
    subprocess.run(cmd, shell=True, check=True)
    os.remove(filelist_path)
    return merged_video

def copy_to_outputs(video_path):
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    base = os.path.basename(video_path)
    new_path = os.path.join(outputs_dir, f"input_{base}")
    if not os.path.exists(new_path):
        shutil.copy(video_path, new_path)
    return new_path

def get_common_file(new_path, old_path):
    if os.path.exists(new_path):
        return new_path
    elif os.path.exists(old_path):
        return old_path
    else:
        print(f"[WARNING] Neither {new_path} nor {old_path} found. Using {old_path} as fallback.")
        return old_path

# ------------------------- Pipeline Management Helpers -------------------------

def has_model_config_changed(old_config, new_config):
    critical_keys = ["model_choice", "torch_dtype", "num_persistent"]
    for key in critical_keys:
        if str(old_config.get(key)) != str(new_config.get(key)):
            print(f"[CMD - DEBUG] Critical config change detected in {key}: {old_config.get(key)} != {new_config.get(key)}")
            return True
    
    lora_keys = [
        "lora_model", "lora_alpha",
        "lora_model_2", "lora_alpha_2",
        "lora_model_3", "lora_alpha_3",
        "lora_model_4", "lora_alpha_4"
    ]
    
    for key in lora_keys:
        old_val = old_config.get(key)
        new_val = new_config.get(key)
        if str(old_val).strip() in ["", "None"] and str(new_val).strip() in ["", "None"]:
            continue
        if str(old_val) != str(new_val):
            print(f"[CMD - DEBUG] LoRA config change detected in {key}: {old_val} != {new_val}")
            return True
    return False

def clear_pipeline_if_needed(pipeline, pipeline_config, new_config):
    global model_manager

    print(f"[CMD - DEBUG] Checking if pipeline needs clearing...")

    if pipeline is not None and has_model_config_changed(pipeline_config, new_config):
        print(f"[CMD - DEBUG] Pipeline config changed. Clearing pipeline.")
        try:
            del pipeline
        except Exception as e:
            print(f"[CMD] Error deleting pipeline: {e}")
        pipeline = None
        pipeline_config = {}

        print(f"[CMD - DEBUG] Checking hasattr(model_manager, 'clear_models'): {hasattr(model_manager, 'clear_models')}")
        print(f"[CMD - DEBUG] dir(model_manager): {dir(model_manager)}")

        if model_manager is not None and hasattr(model_manager, 'clear_models'):
            print(f"[CMD - DEBUG] Calling model_manager.clear_models()")
            model_manager.clear_models()
        else:
            print(f"[CMD - DEBUG] model_manager is None OR hasattr(model_manager, 'clear_models') is False. Skipping model_manager.clear_models()")
        try:
            del model_manager
            model_manager = None
        except Exception as e:
            print(f"[CMD] Error deleting model_manager: {e}")

        if model_manager is None:
            print(f"[CMD - DEBUG] Reinitializing model_manager")
            model_manager = ModelManager(device="cpu")

        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

        print("[CMD] Pipeline and Model Manager cleared due to model config change.")
    else:
        print(f"[CMD - DEBUG] Pipeline config not changed or pipeline is None. No clearing needed.")
    return pipeline, pipeline_config

# ------------------------- Configuration Management -------------------------

CONFIG_DIR = "configs"
LAST_CONFIG_FILE = os.path.join(CONFIG_DIR, "last_used_config.txt")
DEFAULT_CONFIG_NAME = "Default"

def get_default_config():
    return {
        "model_choice": "WAN 2.1 1.3B (Text/Video-to-Video)",
        "vram_preset": "24GB",
        "aspect_ratio": "16:9",
        "width": 832,
        "height": 480,
        "auto_crop": True,
        "auto_scale": False,
        "tiled": True,
        "inference_steps": 50,
        "pr_rife": True,
        "pr_rife_multiplier": "2x FPS",
        "cfg_scale": 5.0,
        "sigma_shift": 5.6,
        "num_persistent": "12000000000",
        "torch_dtype": "torch.bfloat16",
        "lora_model": "None",
        "lora_alpha": 1.0,
        "lora_model_2": "None",
        "lora_alpha_2": 1.0,
        "lora_model_3": "None",
        "lora_alpha_3": 1.0,
        "lora_model_4": "None",
        "lora_alpha_4": 1.0,
        "clear_cache_after_gen": DEFAULT_CLEAR_CACHE,
        "prompt": "",
        "negative_prompt": "Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down",
        "save_prompt": True,
        "multiline": False,
        "num_generations": 1,
        "use_random_seed": True,
        "seed": "",
        "quality": 5,
        "fps": 16,
        "num_frames": 81,
        "denoising_strength": 0.7,
        "tar_lang": "EN",
        "batch_folder": "batch_inputs",
        "batch_output_folder": "batch_outputs",
        "skip_overwrite": True,
        "save_prompt_batch": True,
        "enable_teacache": False,
        "tea_cache_l1_thresh": 0.15,
        "tea_cache_model_id": "Wan2.1-T2V-1.3B",
        "extend_factor": 1
    }

if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

default_config = get_default_config()

if os.path.exists(LAST_CONFIG_FILE):
    with open(LAST_CONFIG_FILE, "r", encoding="utf-8") as f:
        last_config_name = f.read().strip()
    config_file_path = os.path.join(CONFIG_DIR, f"{last_config_name}.json")
    if os.path.exists(config_file_path):
        with open(config_file_path, "r", encoding="utf-8") as f:
            config_loaded = json.load(f)
        last_config = last_config_name
    else:
        default_config_path = os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")
        if os.path.exists(default_config_path):
            with open(default_config_path, "r", encoding="utf-8") as f:
                config_loaded = json.load(f)
            last_config = DEFAULT_CONFIG_NAME
        else:
            config_loaded = default_config
            with open(default_config_path, "w", encoding="utf-8") as f:
                json.dump(config_loaded, f, indent=4)
            last_config = DEFAULT_CONFIG_NAME
            with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
                f.write(last_config)
else:
    default_config_path = os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")
    if os.path.exists(default_config_path):
        with open(default_config_path, "r", encoding="utf-8") as f:
            config_loaded = json.load(f)
        last_config = DEFAULT_CONFIG_NAME
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_NAME)
    else:
        config_loaded = default_config
        with open(default_config_path, "w", encoding="utf-8") as f:
            json.dump(config_loaded, f, indent=4)
        last_config = DEFAULT_CONFIG_NAME
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_NAME)

def get_config_list():
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
    files = os.listdir(CONFIG_DIR)
    configs = [os.path.splitext(f)[0] for f in files if f.endswith(".json")]
    return sorted(configs)

def save_config(config_name, model_choice, vram_preset, aspect_ratio, width, height, auto_crop, auto_scale, tiled, inference_steps,
                pr_rife, pr_rife_multiplier, cfg_scale, sigma_shift, num_persistent, torch_dtype, lora_model, lora_alpha,
                lora_model_2, lora_alpha_2, lora_model_3, lora_alpha_3, lora_model_4, lora_alpha_4, clear_cache_after_gen,
                negative_prompt, save_prompt, multiline, num_generations, use_random_seed, seed, quality, fps, num_frames,
                denoising_strength, tar_lang, batch_folder, batch_output_folder, skip_overwrite, save_prompt_batch,
                enable_teacache, tea_cache_l1_thresh, tea_cache_model_id, extend_factor, prompt):
    if not config_name:
        return "Config name cannot be empty", gr.update(choices=get_config_list())
    
    prompt = str(prompt) if prompt is not None else ""
    
    config_data = {
        "model_choice": model_choice,
        "vram_preset": vram_preset,
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "auto_crop": auto_crop,
        "auto_scale": auto_scale,
        "tiled": tiled,
        "inference_steps": inference_steps,
        "pr_rife": pr_rife,
        "pr_rife_multiplier": pr_rife_multiplier,
        "cfg_scale": cfg_scale,
        "sigma_shift": sigma_shift,
        "num_persistent": num_persistent,
        "torch_dtype": torch_dtype,
        "lora_model": lora_model,
        "lora_alpha": lora_alpha,
        "lora_model_2": lora_model_2,
        "lora_alpha_2": lora_alpha_2,
        "lora_model_3": lora_model_3,
        "lora_alpha_3": lora_alpha_3,
        "lora_model_4": lora_model_4,
        "lora_alpha_4": lora_alpha_4,
        "clear_cache_after_gen": clear_cache_after_gen,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "save_prompt": save_prompt,
        "multiline": multiline,
        "num_generations": num_generations,
        "use_random_seed": use_random_seed,
        "seed": seed,
        "quality": quality,
        "fps": fps,
        "num_frames": num_frames,
        "denoising_strength": denoising_strength,
        "tar_lang": tar_lang,
        "batch_folder": batch_folder,
        "batch_output_folder": batch_output_folder,
        "skip_overwrite": skip_overwrite,
        "save_prompt_batch": save_prompt_batch,
        "enable_teacache": enable_teacache,
        "tea_cache_l1_thresh": tea_cache_l1_thresh,
        "tea_cache_model_id": tea_cache_model_id,
        "extend_factor": extend_factor
    }
    
    try:
        config_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(config_name)
        
        global last_selected_aspect_ratio
        last_selected_aspect_ratio = aspect_ratio
        
        return (
            f"Config '{config_name}' saved and loaded.",
            gr.update(choices=get_config_list(), value=config_name),
            model_choice,
            vram_preset,
            aspect_ratio,
            width,
            height,
            auto_crop,
            auto_scale,
            tiled,
            inference_steps,
            pr_rife,
            pr_rife_multiplier,
            cfg_scale,
            sigma_shift,
            num_persistent,
            torch_dtype,
            lora_model,
            lora_alpha,
            lora_model_2,
            lora_alpha_2,
            lora_model_3,
            lora_alpha_3,
            lora_model_4,
            lora_alpha_4,
            clear_cache_after_gen,
            negative_prompt,
            save_prompt,
            multiline,
            num_generations,
            use_random_seed,
            seed,
            quality,
            fps,
            num_frames,
            denoising_strength,
            tar_lang,
            batch_folder,
            batch_output_folder,
            skip_overwrite,
            save_prompt_batch,
            enable_teacache,
            tea_cache_l1_thresh,
            tea_cache_model_id,
            extend_factor,
            prompt
        )
    except Exception as e:
        return (
            f"Error saving config: {str(e)}",
            gr.update(choices=get_config_list()),
            model_choice,
            vram_preset,
            aspect_ratio,
            width,
            height,
            auto_crop,
            auto_scale,
            tiled,
            inference_steps,
            pr_rife,
            pr_rife_multiplier,
            cfg_scale,
            sigma_shift,
            num_persistent,
            torch_dtype,
            lora_model,
            lora_alpha,
            lora_model_2,
            lora_alpha_2,
            lora_model_3,
            lora_alpha_3,
            lora_model_4,
            lora_alpha_4,
            clear_cache_after_gen,
            negative_prompt,
            save_prompt,
            multiline,
            num_generations,
            use_random_seed,
            seed,
            quality,
            fps,
            num_frames,
            denoising_strength,
            tar_lang,
            batch_folder,
            batch_output_folder,
            skip_overwrite,
            save_prompt_batch,
            enable_teacache,
            tea_cache_l1_thresh,
            tea_cache_model_id,
            extend_factor,
            prompt
        )

def load_config(selected_config):
    global last_selected_aspect_ratio
    
    default_vals = get_default_config()
    if not os.path.exists(os.path.join(CONFIG_DIR, f"{selected_config}.json")):
        return (
            f"Config '{selected_config}' not found.",
            default_vals["model_choice"],
            default_vals["vram_preset"],
            default_vals["aspect_ratio"],
            default_vals["width"],
            default_vals["height"],
            default_vals["auto_crop"],
            default_vals["auto_scale"],
            default_vals["tiled"],
            default_vals["inference_steps"],
            default_vals["pr_rife"],
            default_vals["pr_rife_multiplier"],
            default_vals["cfg_scale"],
            default_vals["sigma_shift"],
            default_vals["num_persistent"],
            default_vals["torch_dtype"],
            default_vals["lora_model"],
            default_vals["lora_alpha"],
            default_vals["lora_model_2"],
            default_vals["lora_alpha_2"],
            default_vals["lora_model_3"],
            default_vals["lora_alpha_3"],
            default_vals["lora_model_4"],
            default_vals["lora_alpha_4"],
            default_vals["clear_cache_after_gen"],
            default_vals["negative_prompt"],
            default_vals["save_prompt"],
            default_vals["multiline"],
            default_vals["num_generations"],
            default_vals["use_random_seed"],
            default_vals["seed"],
            default_vals["quality"],
            default_vals["fps"],
            default_vals["num_frames"],
            default_vals["denoising_strength"],
            default_vals["tar_lang"],
            default_vals["batch_folder"],
            default_vals["batch_output_folder"],
            default_vals["skip_overwrite"],
            default_vals["save_prompt_batch"],
            default_vals["enable_teacache"],
            default_vals["tea_cache_l1_thresh"],
            default_vals["tea_cache_model_id"],
            default_vals["extend_factor"],
            default_vals["prompt"]
        )
    
    try:
        with open(os.path.join(CONFIG_DIR, f"{selected_config}.json"), "r", encoding="utf-8") as f:
            config_data = json.load(f)
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(selected_config)
        
        last_selected_aspect_ratio = config_data.get("aspect_ratio", "16:9")
        
        return (
            f"Config '{selected_config}' loaded.",
            config_data.get("model_choice", "WAN 2.1 1.3B (Text/Video-to-Video)"),
            config_data.get("vram_preset", "24GB"),
            config_data.get("aspect_ratio", "16:9"),
            config_data.get("width", 832),
            config_data.get("height", 480),
            config_data.get("auto_crop", True),
            config_data.get("auto_scale", False),
            config_data.get("tiled", True),
            config_data.get("inference_steps", 50),
            config_data.get("pr_rife", True),
            config_data.get("pr_rife_multiplier", "2x FPS"),
            config_data.get("cfg_scale", 5.0),
            config_data.get("sigma_shift", 5.6),
            config_data.get("num_persistent", "12000000000"),
            config_data.get("torch_dtype", "torch.bfloat16"),
            config_data.get("lora_model", "None"),
            config_data.get("lora_alpha", 1.0),
            config_data.get("lora_model_2", "None"),
            config_data.get("lora_alpha_2", 1.0),
            config_data.get("lora_model_3", "None"),
            config_data.get("lora_alpha_3", 1.0),
            config_data.get("lora_model_4", "None"),
            config_data.get("lora_alpha_4", 1.0),
            config_data.get("clear_cache_after_gen", DEFAULT_CLEAR_CACHE),
            config_data.get("negative_prompt", "Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down"),
            config_data.get("save_prompt", True),
            config_data.get("multiline", False),
            config_data.get("num_generations", 1),
            config_data.get("use_random_seed", True),
            config_data.get("seed", ""),
            config_data.get("quality", 5),
            config_data.get("fps", 16),
            config_data.get("num_frames", 81),
            config_data.get("denoising_strength", 0.7),
            config_data.get("tar_lang", "EN"),
            config_data.get("batch_folder", "batch_inputs"),
            config_data.get("batch_output_folder", "batch_outputs"),
            config_data.get("skip_overwrite", True),
            config_data.get("save_prompt_batch", True),
            config_data.get("enable_teacache", False),
            config_data.get("tea_cache_l1_thresh", 0.15),
            config_data.get("tea_cache_model_id", "Wan2.1-T2V-1.3B"),
            config_data.get("extend_factor", 1),
            config_data.get("prompt", "")
        )
    except Exception as e:
        return (
            f"Error loading config: {str(e)}",
            default_vals["model_choice"],
            default_vals["vram_preset"],
            default_vals["aspect_ratio"],
            default_vals["width"],
            default_vals["height"],
            default_vals["auto_crop"],
            default_vals["auto_scale"],
            default_vals["tiled"],
            default_vals["inference_steps"],
            default_vals["pr_rife"],
            default_vals["pr_rife_multiplier"],
            default_vals["cfg_scale"],
            default_vals["sigma_shift"],
            default_vals["num_persistent"],
            default_vals["torch_dtype"],
            default_vals["lora_model"],
            default_vals["lora_alpha"],
            default_vals["lora_model_2"],
            default_vals["lora_alpha_2"],
            default_vals["lora_model_3"],
            default_vals["lora_alpha_3"],
            default_vals["lora_model_4"],
            default_vals["lora_alpha_4"],
            default_vals["clear_cache_after_gen"],
            default_vals["negative_prompt"],
            default_vals["save_prompt"],
            default_vals["multiline"],
            default_vals["num_generations"],
            default_vals["use_random_seed"],
            default_vals["seed"],
            default_vals["quality"],
            default_vals["fps"],
            default_vals["num_frames"],
            default_vals["denoising_strength"],
            default_vals["tar_lang"],
            default_vals["batch_folder"],
            default_vals["batch_output_folder"],
            default_vals["skip_overwrite"],
            default_vals["save_prompt_batch"],
            default_vals["enable_teacache"],
            default_vals["tea_cache_l1_thresh"],
            default_vals["tea_cache_model_id"],
            default_vals["extend_factor"],
            default_vals["prompt"]
        )

def process_random_prompt(prompt):
    pattern = r'<random:\s*([^>]+)>'
    def replacer(match):
        options = [option.strip() for option in match.group(1).split(',') if option.strip()]
        if options:
            return random.choice(options)
        return ''
    return re.sub(pattern, replacer, prompt)

def compute_auto_scale_dimensions(image, default_width, default_height):
    target_area = default_width * default_height
    orig_w, orig_h = image.size
    if orig_w * orig_h <= target_area:
        return orig_w, orig_h
    scale_factor = (target_area / (orig_w * orig_h)) ** 0.5
    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    new_w = max(new_w, 16)
    new_h = max(new_h, 16)
    return new_w, new_h

def update_target_dimensions(image, auto_scale, current_width, current_height):
    if auto_scale and image is not None:
        try:
            new_w, new_h = compute_auto_scale_dimensions(image, current_width, current_height)
            return new_w, new_h
        except Exception as e:
            return current_width, current_height
    return current_width, current_height

def auto_crop_image(image, target_width, target_height):
    w, h = image.size
    target_ratio = target_width / target_height
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_width = int(h * target_ratio)
        left = (w - new_width) // 2
        right = left + new_width
        image = image.crop((left, 0, right, h))
    elif current_ratio < target_ratio:
        new_height = int(w / target_ratio)
        top = (h - new_height) // 2
        bottom = top + new_height
        image = image.crop((0, top, w, bottom))
    image = image.resize((target_width, target_height), Image.LANCZOS)
    return image

def auto_scale_image(image, target_width, target_height):
    target_area = target_width * target_height
    orig_w, orig_h = image.size
    if orig_w * orig_h <= target_area:
        return image
    scale_factor = (target_area / (orig_w * orig_h)) ** 0.5
    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    new_w = max(new_w, 16)
    new_h = max(new_h, 16)
    return image.resize((new_w, new_h), Image.LANCZOS)

def toggle_lora_visibility(current_visibility):
    new_visibility = not current_visibility
    new_label = "Hide More LoRAs" if new_visibility else "Show More LoRAs"
    return gr.update(visible=new_visibility), new_visibility, new_label

def update_tea_cache_model_id(model_choice):
    if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)":
        return "Wan2.1-T2V-1.3B"
    elif model_choice == "WAN 2.1 14B Text-to-Video":
        return "Wan2.1-T2V-14B"
    elif model_choice == "WAN 2.1 14B Image-to-Video 720P":
        return "Wan2.1-I2V-14B-720P"
    elif model_choice == "WAN 2.1 14B Image-to-Video 480P":
        return "Wan2.1-I2V-14B-480P"
    return "Wan2.1-T2V-1.3B"

ASPECT_RATIOS_1_3b = {
    "1:1":  (640, 640),
    "4:3":  (736, 544),
    "3:4":  (544, 736),
    "3:2":  (768, 512),
    "2:3":  (512, 768),
    "16:9": (832, 480),
    "9:16": (480, 832),
    "21:9": (960, 416),
    "9:21": (416, 960),
    "4:5":  (560, 704),
    "5:4":  (704, 560),
}

ASPECT_RATIOS_14b = {
    "1:1":  (960, 960),
    "4:3":  (1104, 832),
    "3:4":  (832, 1104),
    "3:2":  (1152, 768),
    "2:3":  (768, 1152),
    "16:9": (1280, 720),
    "16:9_low": (832, 480),
    "9:16": (720, 1280),
    "9:16_low": (480, 832),
    "21:9": (1472, 624),
    "9:21": (624, 1472),
    "4:5":  (864, 1072),
    "5:4":  (1072, 864),
}

def update_vram_and_resolution(model_choice, preset, torch_dtype):
    print(model_choice)
    if torch_dtype == "torch.float8_e4m3fn":
        if model_choice == "WAN 2.1 14B Text-to-Video":
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "0",
                "16GB": "0",
                "24GB": "8,750,000,000",
                "32GB": "22,000,000,000",
                "48GB": "22,000,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_14b.keys())
            default_aspect = "16:9"
        elif model_choice == "WAN 2.1 14B Image-to-Video 720P":
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "0",
                "16GB": "0",
                "24GB": "6,000,000,000",
                "32GB": "14,000,000,000",
                "48GB": "22,000,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_14b.keys())
            default_aspect = "16:9"
        elif model_choice == "WAN 2.1 14B Image-to-Video 480P":
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "2,500,000,000",
                "16GB": "7,500,000,000",
                "24GB": "15,000,000,000",
                "32GB": "22,000,000,000",
                "48GB": "22,000,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_1_3b.keys())
            default_aspect = "16:9"
        else:
            if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)":
                mapping = {
                    "4GB": "0",
                    "6GB": "500,000,000",
                    "8GB": "7,000,000,000",
                    "10GB": "7,000,000,000",
                    "12GB": "7,000,000,000",
                    "16GB": "7,000,000,000",
                    "24GB": "7,000,000,000",
                    "32GB": "7,000,000,000",
                    "48GB": "7,000,000,000",
                    "80GB": "7,000,000,000"
                }
                resolution_choices = list(ASPECT_RATIOS_1_3b.keys())
                default_aspect = "16:9"
            else:
                mapping = {
                    "4GB": "0",
                    "6GB": "0",
                    "8GB": "0",
                    "10GB": "0",
                    "12GB": "0",
                    "16GB": "0",
                    "24GB": "3,000,000,000",
                    "32GB": "6,500,000,000",
                    "48GB": "16,000,000,000",
                    "80GB": "22,000,000,000"
                }
                resolution_choices = list(ASPECT_RATIOS_14b.keys())
                default_aspect = "16:9"
        return mapping.get(preset, "12000000000"), resolution_choices, default_aspect
    else:
        if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)":
            mapping = {
                "4GB": "0",
                "6GB": "500,000,000",
                "8GB": "7,000,000,000",
                "10GB": "7,000,000,000",
                "12GB": "7,000,000,000",
                "16GB": "7,000,000,000",
                "24GB": "7,000,000,000",
                "32GB": "7,000,000,000",
                "48GB": "7,000,000,000",
                "80GB": "7,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_1_3b.keys())
            default_aspect = "16:9"
        elif model_choice == "WAN 2.1 14B Text-to-Video":
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "0",
                "16GB": "0",
                "24GB": "4,250,000,000",
                "32GB": "6,500,000,000",
                "48GB": "22,000,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_14b.keys())
            default_aspect = "16:9"
        elif model_choice == "WAN 2.1 14B Image-to-Video 720P":
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "0",
                "16GB": "0",
                "24GB": "3,000,000,000",
                "32GB": "5,500,000,000",
                "48GB": "14,500,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_14b.keys())
            default_aspect = "16:9"
        elif model_choice == "WAN 2.1 14B Image-to-Video 480P":
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "1,500,000,000",
                "16GB": "3,500,000,000",
                "24GB": "7,000,000,000",
                "32GB": "10,500,000,000",
                "48GB": "22,000,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_1_3b.keys())
            default_aspect = "16:9"
        else:
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "0",
                "16GB": "0",
                "24GB": "3,000,000,000",
                "32GB": "5,500,000,000",
                "48GB": "16,000,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_14b.keys())
            default_aspect = "16:9"
        return mapping.get(preset, "12000000000"), resolution_choices, default_aspect

def update_model_settings(model_choice, current_vram_preset, torch_dtype):
    global last_selected_aspect_ratio
    
    num_persistent_val, aspect_options, default_aspect = update_vram_and_resolution(model_choice, current_vram_preset, torch_dtype)
    
    aspect_to_use = last_selected_aspect_ratio if last_selected_aspect_ratio else default_aspect
    
    # The issue is here - we need to preserve the low aspect ratio even when switching models
    # If the aspect ratio includes "_low" suffix, we should try to find the base aspect ratio
    if aspect_to_use and "_low" in aspect_to_use:
        base_aspect = aspect_to_use.split("_")[0]  # Extract base aspect ratio without "_low"
        
        # If switching to 1.3B model from 14B with a low aspect ratio
        if (model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)" or model_choice == "WAN 2.1 14B Image-to-Video 480P"):
            # Check if base aspect exists in 1.3B options
            if base_aspect in ASPECT_RATIOS_1_3b:
                aspect_to_use = base_aspect
            else:
                aspect_to_use = default_aspect
        # Keep the "_low" version when using 14B models
        else:
            # If the low variant isn't in choices, fall back to base aspect
            if aspect_to_use not in ASPECT_RATIOS_14b:
                if base_aspect in ASPECT_RATIOS_14b:
                    aspect_to_use = base_aspect
                else:
                    aspect_to_use = default_aspect
    else:
        # Original logic for non-low aspect ratios
        if (model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)" or model_choice == "WAN 2.1 14B Image-to-Video 480P"):
            if aspect_to_use not in ASPECT_RATIOS_1_3b:
                aspect_to_use = default_aspect
            default_width, default_height = ASPECT_RATIOS_1_3b.get(aspect_to_use, (832, 480))
        else:
            if aspect_to_use not in ASPECT_RATIOS_14b:
                aspect_to_use = default_aspect
            default_width, default_height = ASPECT_RATIOS_14b.get(aspect_to_use, (1280, 720))
    
    # Get width and height based on selected aspect ratio
    if (model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)" or model_choice == "WAN 2.1 14B Image-to-Video 480P"):
        default_width, default_height = ASPECT_RATIOS_1_3b.get(aspect_to_use, (832, 480))
    else:
        default_width, default_height = ASPECT_RATIOS_14b.get(aspect_to_use, (1280, 720))
    
    return (
        gr.update(choices=aspect_options, value=aspect_to_use),
        default_width,
        default_height,
        num_persistent_val
    )

def update_width_height(aspect_ratio, model_choice):
    global last_selected_aspect_ratio
    if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)" or model_choice == "WAN 2.1 14B Image-to-Video 480P":
        # For 1.3B models, we need to remove "_low" suffix if present, since it's only valid for 14B models
        if aspect_ratio and "_low" in aspect_ratio:
            base_aspect = aspect_ratio.split("_")[0]
            if base_aspect in ASPECT_RATIOS_1_3b:
                aspect_ratio = base_aspect
            else:
                aspect_ratio = "16:9"  # fallback
        elif aspect_ratio not in ASPECT_RATIOS_1_3b:
            aspect_ratio = "16:9"  # fallback to default if invalid for 1.3B
    else:
        # For 14B models, preserve the low aspect ratio if it exists
        if aspect_ratio not in ASPECT_RATIOS_14b:
            # If it has "_low" suffix but not in dictionary
            if aspect_ratio and "_low" in aspect_ratio:
                base_aspect = aspect_ratio.split("_")[0]
                if base_aspect in ASPECT_RATIOS_14b:
                    aspect_ratio = base_aspect
                else:
                    aspect_ratio = "16:9"  # fallback
            else:
                aspect_ratio = "16:9"  # fallback
    
    last_selected_aspect_ratio = aspect_ratio
    
    if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)" or model_choice == "WAN 2.1 14B Image-to-Video 480P":
        default_width, default_height = ASPECT_RATIOS_1_3b.get(aspect_ratio, (832, 480))
    else:
        default_width, default_height = ASPECT_RATIOS_14b.get(aspect_ratio, (1280, 720))
    
    return default_width, default_height

def update_vram_on_change(preset, model_choice):
    torch_dtype = config_loaded.get("torch_dtype", "torch.bfloat16")
    num_persistent_val, _, _ = update_vram_and_resolution(model_choice, preset, torch_dtype)
    return num_persistent_val

def prompt_enc(prompt, tar_lang):
    global prompt_expander, loaded_pipeline, loaded_pipeline_config, args
    if prompt_expander is None:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(model_name=args.prompt_extend_model, is_vl=False)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(model_name=args.prompt_extend_model, is_vl=False, device=0)
        else:
            raise NotImplementedError(f"Unsupported prompt_extend_method: {args.prompt_extend_method}")
    prompt_output = prompt_expander(prompt, tar_lang=tar_lang.lower())
    result = prompt if not prompt_output.status else prompt_output.prompt
    return result

def show_extension_info():
    info = (
        "**扩展视频功能–详细说明：**\n\n"
        "**说明:**\n"
        "- “扩展视频”滑块允许您通过添加额外片段来自动延长生成的视频。这些片段是使用前一片段的最后一帧生成的，以保持连续性。\n\n"
        "**工作原理:**\n"
        "1. **第一代：**\n"
        "   - 如果滑块设置为1×，则不应用扩展，则只生成一个视频片段。.\n"
        "   - 对于大于1×的值，应用程序首先正常生成初始视频片段。\n\n"
        "2. **确定分段数量：**\n"
        "   - 对于图像输入：分段总数等于滑块值减1。.\n"
        "   - 对于视频输入：适用类似的逻辑（内部进行一些调整）\n\n"
        "3. **扩展生成过程：**\n"
        "   - 对于每一个额外的扩展，该应用程序都会使用OpenCV从最近的片段中提取最后一帧。.\n"
        "   - 然后，使用几乎相同的参数将最后一帧重新馈送到生成管道，以确保一致性。\n\n"
        "4. **合并分段：**\n"
        "   - 所有生成的片段，包括基本片段和所有扩展，都使用ffmpeg合并成一个最终的视频。\n\n"
        "5. **可选帧率增强：**\n"
        "   - 如果启用了实用RIFE选项，最终合并的视频将进行帧率增强，以实现更平滑的运动。\n\n"
        "6. **批量处理：**\n"
        "   - 处理文件文件夹时，会应用类似的扩展过程。\n\n"
        "点击此按钮，您可以全面了解应用程序在幕后的工作。"
    )
    return info

# ------------------------- Single Generation Pipeline (Improved) -------------------------

def get_next_generation_number(output_folder):
    import re
    max_num = 0
    if os.path.exists(output_folder):
        for f in os.listdir(output_folder):
            m = re.match(r'^(\d{4})\.mp4$', f)
            if m:
                num = int(m.group(1))
                if num > max_num:
                    max_num = num
    return max_num + 1

def generate_videos(
    prompt, tar_lang, negative_prompt, input_image, input_video, denoising_strength, num_generations,
    save_prompt, multi_line, use_random_seed, seed_input, quality, fps,
    model_choice_radio, vram_preset, num_persistent_input, torch_dtype, num_frames,
    aspect_ratio, width, height, auto_crop, auto_scale, tiled,
    inference_steps, pr_rife_enabled, pr_rife_radio, cfg_scale, sigma_shift,
    enable_teacache, tea_cache_l1_thresh, tea_cache_model_id,
    lora_model, lora_alpha,
    lora_model_2, lora_alpha_2,
    lora_model_3, lora_alpha_slider_3,
    lora_model_4, lora_alpha_4,
    clear_cache_after_gen, extend_factor,
    override_input_file=None,
    output_dir_override="outputs",
    custom_output_filename=None
):
    global loaded_pipeline, loaded_pipeline_config, cancel_flag, prompt_expander

    output_folder = output_dir_override
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    if input_image is None and input_video is None and override_input_file is not None:
        ext = os.path.splitext(override_input_file)[1].lower()
        if ext == ".mp4":
            input_video = override_input_file
        else:
            try:
                loaded_img = Image.open(override_input_file)
                loaded_img = ImageOps.exif_transpose(loaded_img)
                input_image = loaded_img.convert("RGB")
            except Exception as e:
                print(f"[CMD] Error loading file {override_input_file}: {e}")

    cancel_flag = False
    log_text = ""
    last_used_seed = None
    overall_start_time = time.time()
    final_output_video = None

    input_was_video = False
    orig_video_path = None
    copied_input_video = None
    if input_image is None and input_video is not None:
        input_was_video = True
        orig_video_path = input_video if isinstance(input_video, str) else input_video.name
        copied_input_video = copy_to_outputs(orig_video_path)
        log_text += f"[CMD] Copied input video to: {copied_input_video}\n"

    if model_choice_radio == "WAN 2.1 1.3B (Text/Video-to-Video)":
        model_choice = "1.3B"
        d = ASPECT_RATIOS_1_3b
    elif model_choice_radio == "WAN 2.1 14B Text-to-Video":
        model_choice = "14B_text"
        d = ASPECT_RATIOS_14b
    elif model_choice_radio == "WAN 2.1 14B Image-to-Video 720P":
        model_choice = "14B_image_720p"
        d = ASPECT_RATIOS_14b
    elif model_choice_radio == "WAN 2.1 14B Image-to-Video 480P":
        model_choice = "14B_image_480p"
        d = ASPECT_RATIOS_1_3b
    else:
        return None, "Invalid model choice.", ""
    
    target_width = int(width)
    target_height = int(height)

    if model_choice in ["14B_image_720p", "14B_image_480p"]:
        if input_image is None:
            if input_video is not None:
                video_path = input_video if isinstance(input_video, str) else input_video.name
                original_image = extract_last_frame(video_path)
                if original_image is None:
                    err_msg = "[CMD] Error: Could not extract image from provided video. Please upload a valid input image."
                    if clear_cache_after_gen:
                        loaded_pipeline = None
                        loaded_pipeline_config = {}
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    return None, err_msg, str(last_used_seed or "")
                log_text += "[CMD] Extracted last frame from input video for image-to-video generation.\n"
            else:
                err_msg = "[CMD] Error: Image model selected but no image provided. Please upload input image."
                if clear_cache_after_gen:
                    loaded_pipeline = None
                    loaded_pipeline_config = {}
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return None, err_msg, str(last_used_seed or "")
        else:
            original_image = input_image.copy()
    elif auto_crop or auto_scale:
        if input_image is not None:
            original_image = input_image.copy()
        else:
            original_image = None

    if model_choice == "1.3B" and input_video is not None:
        original_video_path = input_video if isinstance(input_video, str) else input_video.name
        cap = cv2.VideoCapture(original_video_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            effective_num_frames = min(int(num_frames), total_frames)
            print(f"[CMD] Detected input video frame count: {total_frames}, using effective frame count: {effective_num_frames}")
        else:
            effective_num_frames = int(num_frames)
            print("[CMD] Could not open input video, using provided frame count")
        cap.release()
    else:
        effective_num_frames = int(num_frames)

    num_persistent_input = str(num_persistent_input).replace(",", "")
    vram_value = int(num_persistent_input)
    effective_loras = []
    if lora_model and lora_model != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model), lora_alpha))
    if lora_model_2 and lora_model_2 != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model_2), lora_alpha_2))
    if lora_model_3 and lora_model_3 != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model_3), lora_alpha_slider_3))
    if lora_model_4 and lora_model_4 != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model_4), lora_alpha_4))
        
    # Define a helper to format LoRA alpha values consistently
    def format_alpha(alpha):
        try:
            return str(float(alpha))
        except Exception:
            return str(alpha)
    
    new_config = {
         "model_choice": model_choice,
         "torch_dtype": torch_dtype,
         "num_persistent": str(vram_value),
         "lora_model": lora_model,
         "lora_alpha": format_alpha(lora_alpha) if lora_model != "None" else "None",
         "lora_model_2": lora_model_2,
         "lora_alpha_2": format_alpha(lora_alpha_2) if lora_model_2 != "None" else "None",
         "lora_model_3": lora_model_3,
         "lora_alpha_3": format_alpha(lora_alpha_slider_3) if lora_model_3 != "None" else "None",
         "lora_model_4": lora_model_4,
         "lora_alpha_4": format_alpha(lora_alpha_4) if lora_model_4 != "None" else "None",
    }
    loaded_pipeline, loaded_pipeline_config = clear_pipeline_if_needed(loaded_pipeline, loaded_pipeline_config, new_config)
    if loaded_pipeline is None:
         loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_value, lora_path=effective_loras, lora_alpha=None)
         loaded_pipeline_config = new_config

    if multi_line:
        prompts_list = [line.strip() for line in prompt.splitlines() if line.strip()]
    else:
        prompts_list = [prompt.strip()]
        
    total_iterations = len(prompts_list) * int(num_generations)
    if custom_output_filename:
        base_name_prefix = custom_output_filename
        counter = 0
    else:
        counter = get_next_generation_number(output_folder)
    
    if use_random_seed:
        base_seed = None
    else:
        try:
            base_seed = int(seed_input.strip()) if seed_input.strip() != "" else random.randint(0, 2**32 - 1)
        except:
            base_seed = random.randint(0, 2**32 - 1)

    for p in prompts_list:
        for gen in range(int(num_generations)):
            if cancel_flag:
                log_text += "[CMD] Generation cancelled by user before starting a new video.\n"
                return None, log_text, str(last_used_seed or "")
                
            if use_random_seed:
                current_seed = random.randint(0, 2**32 - 1)
            else:
                current_seed = base_seed + gen if int(num_generations) > 1 else base_seed
            last_used_seed = current_seed

            if custom_output_filename:
                base_name = f"{base_name_prefix}{'' if counter == 0 else '_' + str(counter)}"
                counter += 1
            else:
                base_name = f"{counter:04d}"
                counter += 1

            log_text += f"[CMD] Generation with prompt: {p} and seed: {current_seed}\n"

            base_config = {
                "model_choice": model_choice,
                "torch_dtype": torch_dtype,
                "num_persistent": str(vram_value),
                "lora_model": lora_model,
                "lora_alpha": format_alpha(lora_alpha) if lora_model != "None" else "None",
                "lora_model_2": lora_model_2,
                "lora_alpha_2": format_alpha(lora_alpha_2) if lora_model_2 != "None" else "None",
                "lora_model_3": lora_model_3,
                "lora_alpha_3": format_alpha(lora_alpha_slider_3) if lora_model_3 != "None" else "None",
                "lora_model_4": lora_model_4,
                "lora_alpha_4": format_alpha(lora_alpha_4) if lora_model_4 != "None" else "None"
            }
            if loaded_pipeline is None or loaded_pipeline_config.get("model_choice") != model_choice:
                loaded_pipeline, loaded_pipeline_config = clear_pipeline_if_needed(loaded_pipeline, loaded_pipeline_config, base_config)
                if loaded_pipeline is None:
                    loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_value, lora_path=effective_loras, lora_alpha=None)
                    loaded_pipeline_config = base_config

            common_args = {
                "prompt": process_random_prompt(p),
                "negative_prompt": negative_prompt,
                "num_inference_steps": int(inference_steps),
                "seed": current_seed,
                "tiled": tiled,
                "width": target_width,
                "height": target_height,
                "num_frames": effective_num_frames,
                "cfg_scale": cfg_scale,
                "sigma_shift": sigma_shift,
            }
            if enable_teacache:
                common_args["tea_cache_l1_thresh"] = tea_cache_l1_thresh
                common_args["tea_cache_model_id"] = tea_cache_model_id
            else:
                common_args["tea_cache_l1_thresh"] = None
                common_args["tea_cache_model_id"] = ""
            
            original_filename = os.path.join(output_folder, f"{base_name}.mp4")
            video_start_time = time.time()

            if model_choice == "1.3B":
                if input_video is not None:
                    video_obj = VideoData(input_video if isinstance(input_video, str) else input_video.name, height=target_height, width=target_width)
                    video_data = loaded_pipeline(
                        input_video=video_obj,
                        denoising_strength=denoising_strength,
                        **common_args,
                        cancel_fn=lambda: cancel_flag
                    )
                else:
                    video_data = loaded_pipeline(
                        **common_args,
                        cancel_fn=lambda: cancel_flag
                    )
            elif model_choice in ["14B_text"]:
                video_data = loaded_pipeline(
                    **common_args,
                    cancel_fn=lambda: cancel_flag
                )
            elif model_choice in ["14B_image_720p", "14B_image_480p"]:
                if auto_crop:
                    processed_image = auto_crop_image(original_image, target_width, target_height)
                elif auto_scale:
                    processed_image = auto_scale_image(original_image, target_width, target_height)
                else:
                    processed_image = original_image

                pre_processed_dir = "auto_pre_processed_images"
                if not os.path.exists(pre_processed_dir):
                    os.makedirs(pre_processed_dir)
                save_filename = os.path.join(pre_processed_dir, f"auto_processed_{int(time.time())}.png")
                try:
                    processed_image.save(save_filename)
                    print(f"[CMD] Auto processed image saved to: {save_filename}")
                except Exception as e:
                    print(f"[CMD] Failed to save auto processed image: {e}")

                video_data = loaded_pipeline(
                    input_image=processed_image,
                    **common_args,
                    cancel_fn=lambda: cancel_flag
                )
            else:
                err_msg = "[CMD] Invalid combination of inputs."
                if clear_cache_after_gen:
                    loaded_pipeline = None
                    loaded_pipeline_config = {}
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return None, err_msg, str(last_used_seed or "")

            video_duration = time.time() - video_start_time
            log_text += f"[CMD] Original video generation duration: {video_duration:.2f} seconds\n"

            save_video(video_data, original_filename, fps=fps, quality=quality)
            log_text += f"[CMD] Saved original video: {original_filename}\n"
            if save_prompt:
                txt_filename = os.path.splitext(original_filename)[0] + ".txt"
                generation_details = generate_prompt_info({
                    "prompt": p,
                    "negative_prompt": negative_prompt,
                    "model_choice": model_choice_radio,
                    "inference_steps": inference_steps,
                    "cfg_scale": cfg_scale,
                    "sigma_shift": sigma_shift,
                    "seed": current_seed,
                    "num_frames": effective_num_frames,
                    "extend_factor": extend_factor,
                    "num_segments": 1,
                    "extension_segment": 0,
                    "total_extensions": int(extend_factor) - 1,
                    "source_frame": "original",
                    "input_file": orig_video_path if input_was_video else "",
                    "is_video": input_was_video,
                    "has_input_video": input_was_video,
                    "denoising_strength": denoising_strength,
                    "is_text_to_video": model_choice == "14B_text",
                    "lora_details": [f"{os.path.basename(path)} (scale {alpha})" for path, alpha in effective_loras],
                    "enable_teacache": enable_teacache,
                    "tea_cache_l1_thresh": tea_cache_l1_thresh,
                    "tea_cache_model_id": tea_cache_model_id,
                    "torch_dtype": torch_dtype,
                    "auto_crop": auto_crop,
                    "width": target_width,
                    "height": target_height,
                    "video_generation_duration": video_duration,
                    "generation_duration": time.time() - overall_start_time,
                    "include_minutes": True
                })
                with open(txt_filename, "w", encoding="utf-8") as f:
                    f.write(generation_details)
                log_text += f"[CMD] Saved prompt info for original video: {txt_filename}\n"
            
            # Only switch the pipeline to extension mode if extend_factor > 1
            if int(extend_factor) > 1:
                extension_model_choice = model_choice
                if model_choice == "1.3B":
                    extension_model_choice = "14B_image_480p"
                elif model_choice == "14B_text":
                    extension_model_choice = "14B_image_720p"
                if extension_model_choice != model_choice:
                    log_text += f"[CMD] Switching pipeline for extension segments to model {extension_model_choice}\n"
                    new_config_ext = new_config.copy()
                    new_config_ext["model_choice"] = extension_model_choice
                    loaded_pipeline, loaded_pipeline_config = clear_pipeline_if_needed(loaded_pipeline, loaded_pipeline_config, new_config_ext)
                    if loaded_pipeline is None:
                        loaded_pipeline = load_wan_pipeline(extension_model_choice, torch_dtype, vram_value, lora_path=effective_loras, lora_alpha=None)
                        loaded_pipeline_config = new_config_ext

            original_improved = None
            ext_segments = []
            ext_segments_improved = []
            
            additional_extensions = int(extend_factor) - 1
            prev_video = original_filename
            for ext_iter in range(1, additional_extensions + 1):
                if cancel_flag:
                    log_text += "[CMD] Generation cancelled by user during extensions.\n"
                    break
                last_frame = extract_last_frame(prev_video)
                if last_frame is None:
                    log_text += f"[CMD] Failed to extract last frame for extension {ext_iter} from {prev_video}.\n"
                    break
                used_folder = "used_last_frames"
                if not os.path.exists(used_folder):
                    os.makedirs(used_folder)
                last_frame_filename = os.path.join(used_folder, f"{base_name}_ext{ext_iter}_lastframe.png")
                last_frame.save(last_frame_filename)
                log_text += f"[CMD] Saved last frame used for extension {ext_iter}: {last_frame_filename}\n"
                new_width, new_height = last_frame.size
                common_args_ext = {
                    "prompt": process_random_prompt(p),
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": int(inference_steps),
                    "seed": random.randint(0, 2**32 - 1) if use_random_seed else (current_seed),
                    "tiled": tiled,
                    "width": new_width,
                    "height": new_height,
                    "num_frames": int(num_frames),
                    "cfg_scale": cfg_scale,
                    "sigma_shift": sigma_shift,
                }
                if enable_teacache:
                    common_args_ext["tea_cache_l1_thresh"] = tea_cache_l1_thresh
                    common_args_ext["tea_cache_model_id"] = tea_cache_model_id
                else:
                    common_args_ext["tea_cache_l1_thresh"] = None
                    common_args_ext["tea_cache_model_id"] = ""
                    
                extension_filename = os.path.join(output_folder, f"{base_name}_ext{ext_iter}.mp4")
                log_text += f"[CMD] Generating extension segment {ext_iter} using model {extension_model_choice if int(extend_factor)>1 else model_choice}\n"
                try:
                    ext_start_time = time.time()
                    
                    video_data_ext = loaded_pipeline(
                        input_image=last_frame,
                        **common_args_ext,
                        cancel_fn=lambda: cancel_flag
                    )
                    
                    ext_duration = time.time() - ext_start_time
                    log_text += f"[CMD] Extension segment {ext_iter} generation duration: {ext_duration:.2f} seconds\n"
                    
                    if not video_data_ext:
                        log_text += "[CMD] Extension generation returned no data.\n"
                        break
                    save_video(video_data_ext, extension_filename, fps=fps, quality=quality)
                    log_text += f"[CMD] Saved extension segment {ext_iter}: {extension_filename}\n"
                    if save_prompt:
                        txt_filename_ext = os.path.splitext(extension_filename)[0] + ".txt"
                        generation_details_ext = generate_prompt_info({
                            "prompt": p,
                            "negative_prompt": negative_prompt,
                            "model_choice": extension_model_choice if int(extend_factor) > 1 else model_choice_radio,
                            "extension_model": extension_model_choice if int(extend_factor) > 1 and extension_model_choice != model_choice_radio else None,
                            "inference_steps": inference_steps,
                            "cfg_scale": cfg_scale,
                            "sigma_shift": sigma_shift,
                            "seed": common_args_ext["seed"],
                            "num_frames": num_frames,
                            "extend_factor": extend_factor,
                            "num_segments": 1,
                            "extension_segment": ext_iter,
                            "total_extensions": additional_extensions,
                            "source_frame": os.path.basename(prev_video),
                            "input_file": copied_input_video if input_was_video else "",
                            "is_video": input_was_video,
                            "has_input_video": input_was_video,
                            "denoising_strength": denoising_strength,
                            "is_text_to_video": model_choice=="14B_text",
                            "lora_details": [f"{os.path.basename(path)} (scale {alpha})" for path, alpha in effective_loras],
                            "enable_teacache": enable_teacache,
                            "tea_cache_l1_thresh": tea_cache_l1_thresh,
                            "tea_cache_model_id": tea_cache_model_id,
                            "torch_dtype": torch_dtype,
                            "auto_crop": auto_crop,
                            "width": new_width,
                            "height": new_height,
                            "video_generation_duration": ext_duration,
                            "generation_duration": time.time() - overall_start_time,
                            "include_minutes": True
                        })
                        with open(txt_filename_ext, "w", encoding="utf-8") as f:
                            f.write(generation_details_ext)
                        log_text += f"[CMD] Saved prompt info for extension segment {ext_iter}: {txt_filename_ext}\n"
                    ext_segments.append(extension_filename)
                    prev_video = extension_filename
                except Exception as e:
                    log_text += f"[CMD] Error during extension generation: {str(e)}\n"
                    continue
            
            if cancel_flag:
                log_text += "[CMD] Generation cancelled by user, skipping post-processing steps.\n"
                if clear_cache_after_gen:
                    loaded_pipeline = None
                    loaded_pipeline_config = {}
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return original_filename, log_text, str(last_used_seed or "")
                
            if pr_rife_enabled:
                if cancel_flag:
                    log_text += "[CMD] Generation cancelled by user before Practical-RIFE processing.\n"
                else:
                    original_improved = os.path.join(output_folder, f"{base_name}_improved.mp4")
                    try:
                        cap = cv2.VideoCapture(original_filename)
                        source_fps = cap.get(cv2.CAP_PROP_FPS)
                        cap.release()
                        if source_fps <= 29:
                            print(f"[CMD] Applying Practical-RIFE on original {original_filename}")
                            multiplier_val = "2" if pr_rife_radio == "2x FPS" else "4"
                            cmd = f'"{sys.executable}" "Practical-RIFE/inference_video.py" --model="{os.path.abspath(os.path.join("Practical-RIFE", "train_log"))}" --multi={multiplier_val} --video="{original_filename}" --output="{original_improved}"'
                            if not cancel_flag:
                                subprocess.run(cmd, shell=True, check=True, env=os.environ)
                                log_text += f"[CMD] Applied Practical-RIFE on original. Saved as: {original_improved}\n"
                            else:
                                log_text += "[CMD] Generation cancelled by user during Practical-RIFE processing.\n"
                                original_improved = original_filename
                        else:
                            original_improved = original_filename
                            log_text += f"[CMD] Skipped Practical-RIFE on original because source FPS ({source_fps:.2f}) is above threshold.\n"
                    except Exception as e:
                        log_text += f"[CMD] Error applying Practical-RIFE on original: {str(e)}\n"
                        original_improved = original_filename
                    
                    for idx, ext_file in enumerate(ext_segments):
                        if cancel_flag:
                            log_text += f"[CMD] Generation cancelled by user before Practical-RIFE processing on extension {idx+1}.\n"
                            ext_segments_improved.append(ext_file)
                            continue
                            
                        ext_improved = os.path.join(output_folder, f"{base_name}_ext{idx+1}_improved.mp4")
                        try:
                            cap = cv2.VideoCapture(ext_file)
                            source_fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.release()
                            if source_fps <= 29:
                                print(f"[CMD] Applying Practical-RIFE on extension {ext_file}")
                                multiplier_val = "2" if pr_rife_radio == "2x FPS" else "4"
                                cmd = f'"{sys.executable}" "Practical-RIFE/inference_video.py" --model="{os.path.abspath(os.path.join("Practical-RIFE", "train_log"))}" --multi={multiplier_val} --video="{ext_file}" --output="{ext_improved}"'
                                if not cancel_flag:
                                    subprocess.run(cmd, shell=True, check=True, env=os.environ)
                                    log_text += f"[CMD] Applied Practical-RIFE on extension {idx+1}. Saved as: {ext_improved}\n"
                                else:
                                    log_text += f"[CMD] Generation cancelled by user during Practical-RIFE processing on extension {idx+1}.\n"
                                    ext_improved = ext_file
                            else:
                                ext_improved = ext_file
                                log_text += f"[CMD] Skipped Practical-RIFE on extension {idx+1} due to high FPS.\n"
                        except Exception as e:
                            log_text += f"[CMD] Error applying Practical-RIFE on extension {idx+1}: {str(e)}\n"
                            ext_improved = ext_file
                        ext_segments_improved.append(ext_improved)
            else:
                original_improved = original_filename

            merged_original = None
            merged_enhanced = None
            if ext_segments and not cancel_flag:
                try:
                    merge_list = [original_filename] + ext_segments
                    merged_original = os.path.join(output_folder, f"{base_name}_extended_original.mp4")
                    filelist_path = os.path.join(tempfile.gettempdir(), "filelist_original.txt")
                    with open(filelist_path, "w", encoding="utf-8") as f:
                        for vf in merge_list:
                            if os.path.exists(vf):
                                f.write(f"file '{os.path.abspath(vf)}'\n")
                            else:
                                log_text += f"[CMD] Warning: file not found: {vf}\n"
                    if os.path.getsize(filelist_path) > 0 and not cancel_flag:
                        cmd = f'ffmpeg -f concat -safe 0 -i "{filelist_path}" -c copy "{merged_original}"'
                        subprocess.run(cmd, shell=True, check=True)
                        os.remove(filelist_path)
                        log_text += f"[CMD] Merged unenhanced extended video saved as: {merged_original}\n"
                    else:
                        if cancel_flag:
                            log_text += "[CMD] Generation cancelled by user before merging original files.\n"
                        else:
                            log_text += f"[CMD] No valid files to merge for extended original.\n"
                        os.remove(filelist_path)
                except Exception as e:
                    log_text += f"[CMD] Error merging original extensions: {str(e)}\n"
                if pr_rife_enabled and ext_segments_improved and not cancel_flag:
                    try:
                        merge_list_improved = [original_improved] + ext_segments_improved
                        suffix = "_extended_original_enhanced" if len(ext_segments_improved) > 1 else "_extended_enhanced"
                        merged_enhanced = os.path.join(output_folder, f"{base_name}{suffix}.mp4")
                        filelist_path = os.path.join(tempfile.gettempdir(), "filelist_enhanced.txt")
                        with open(filelist_path, "w", encoding="utf-8") as f:
                            for vf in merge_list_improved:
                                if os.path.exists(vf):
                                    f.write(f"file '{os.path.abspath(vf)}'\n")
                                else:
                                    log_text += f"[CMD] Warning: file not found: {vf}\n"
                        if os.path.getsize(filelist_path) > 0 and not cancel_flag:
                            cmd = f'ffmpeg -f concat -safe 0 -i "{filelist_path}" -c copy "{merged_enhanced}"'
                            subprocess.run(cmd, shell=True, check=True)
                            os.remove(filelist_path)
                            log_text += f"[CMD] Merged enhanced extended video saved as: {merged_enhanced}\n"
                        else:
                            if cancel_flag:
                                log_text += "[CMD] Generation cancelled by user before merging enhanced files.\n"
                            else:
                                log_text += f"[CMD] No valid files to merge for extended enhanced video.\n"
                            os.remove(filelist_path)
                    except Exception as e:
                        log_text += f"[CMD] Error merging enhanced extensions: {str(e)}\n"

            if pr_rife_enabled and merged_enhanced and os.path.exists(merged_enhanced):
                final_output_video = merged_enhanced
            elif merged_original and os.path.exists(merged_original):
                final_output_video = merged_original
            elif original_improved and os.path.exists(original_improved):
                final_output_video = original_improved
            else:
                final_output_video = original_filename

            log_text += f"[CMD] Completed generation for base {base_name}.\n"
            
            if clear_cache_after_gen:
                loaded_pipeline = None
                loaded_pipeline_config = {}
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    overall_duration = time.time() - overall_start_time
    log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
    log_text += f"[CMD] Generation complete. Overall Duration: {overall_duration:.2f} seconds ({overall_duration/60:.2f} minutes). Last used seed: {last_used_seed}\n"
    print(f"[CMD] Generation complete. Overall Duration: {overall_duration:.2f} seconds. Last used seed: {last_used_seed}")
    
    if clear_cache_after_gen:
        loaded_pipeline = None
        loaded_pipeline_config = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    if final_output_video and os.path.exists(final_output_video):
        return final_output_video, log_text, str(last_used_seed or "")
    elif final_output_video:
        return final_output_video, log_text + "\n[CMD] Warning: Could not generate valid output video.", str(last_used_seed or "")
    else:
        return None, log_text + "\n[CMD] Warning: Could not generate valid output video.", str(last_used_seed or "")

def cancel_generation():
    global cancel_flag, cancel_batch_flag
    cancel_flag = True
    cancel_batch_flag = True
    print("[CMD] Cancel button pressed.")
    return "Cancelling generation..."

# ------------------------- Improved Batch Processing -------------------------

def batch_process_videos(
    default_prompt, folder_path, batch_output_folder, skip_overwrite, tar_lang, negative_prompt, denoising_strength,
    use_random_seed, seed_input, quality, fps, model_choice_radio, vram_preset, num_persistent_input,
    torch_dtype, num_frames, inference_steps, aspect_ratio, width, height, auto_crop, auto_scale,
    tiled, cfg_scale, sigma_shift, save_prompt, pr_rife_enabled, pr_rife_radio, lora_model, lora_alpha,
    lora_model_2, lora_alpha_2, lora_model_3, lora_alpha_3, lora_model_4, lora_alpha_4, enable_teacache,
    tea_cache_l1_thresh, tea_cache_model_id, clear_cache_after_gen, extend_factor, num_generations
):
    global cancel_batch_flag, cancel_flag
    cancel_batch_flag = False
    cancel_flag = False
    log_text = ""
    if not os.path.isdir(folder_path):
        log_text += f"[CMD] Provided folder path does not exist: {folder_path}\n"
        return log_text
    if not os.path.exists(batch_output_folder):
        try:
            os.makedirs(batch_output_folder)
            log_text += f"[CMD] Created batch processing outputs folder: {batch_output_folder}\n"
        except Exception as e:
            log_text += f"[CMD] Error creating output folder {batch_output_folder}: {e}\n"
            return log_text
    files = os.listdir(folder_path)
    allowed_exts = [".jpg", ".png", ".jpeg", ".mp4"]
    files = [f for f in files if os.path.splitext(f)[1].lower() in allowed_exts]
    total_files = len(files)
    log_text += f"[CMD] Found {total_files} files in folder {folder_path}\n"
    for file in files:
        if cancel_batch_flag:
            log_text += "[CMD] Batch processing cancelled by user.\n"
            return log_text
            
        file_path = os.path.join(folder_path, file)
        base, ext = os.path.splitext(file)
        prompt_path = os.path.join(folder_path, base + ".txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_content = f.read().strip()
            if prompt_content == "":
                log_text += f"[CMD] Prompt file {base+'.txt'} is empty, using default prompt.\n"
                prompt_content = default_prompt
            else:
                log_text += f"[CMD] Using prompt from {base+'.txt'} for {file}\n"
        else:
            log_text += f"[CMD] No prompt file for {file}, using default prompt.\n"
            prompt_content = default_prompt
            
        if cancel_batch_flag:
            log_text += "[CMD] Batch processing cancelled by user.\n"
            return log_text
            
        ext_lower = ext.lower()
        if ext_lower == ".mp4":
            image_in = None
            video_in = file_path
        else:
            try:
                loaded_img = Image.open(file_path)
                loaded_img = ImageOps.exif_transpose(loaded_img)
                image_in = loaded_img.convert("RGB")
            except Exception as e:
                log_text += f"[CMD] Error loading image {file_path}: {e}\n"
                continue
            video_in = None
        
        if cancel_batch_flag:
            log_text += "[CMD] Batch processing cancelled by user.\n"
            return log_text
            
        custom_filename = base

        print(f"DEBUG processing {file_path}")
        
        generated_video, single_log, _ = generate_videos(
            prompt_content, tar_lang, negative_prompt, image_in, video_in, denoising_strength, num_generations,
            save_prompt, False, use_random_seed, seed_input, quality, fps,
            model_choice_radio, vram_preset, num_persistent_input, torch_dtype, num_frames,
            aspect_ratio, width, height, auto_crop, auto_scale, tiled, inference_steps, pr_rife_enabled, pr_rife_radio, cfg_scale, sigma_shift,
            enable_teacache, tea_cache_l1_thresh, tea_cache_model_id,
            lora_model, lora_alpha, lora_model_2, lora_alpha_2, lora_model_3, lora_alpha_3, lora_model_4, lora_alpha_4,
            clear_cache_after_gen, extend_factor,
            None,
            output_dir_override=batch_output_folder,
            custom_output_filename=custom_filename
        )
        log_text += single_log
        
        if cancel_batch_flag:
            log_text += "[CMD] Batch processing cancelled by user after file completion.\n"
            return log_text
            
    return log_text

def cancel_batch_process():
    global cancel_batch_flag, cancel_flag
    cancel_batch_flag = True
    cancel_flag = True
    print("[CMD] Batch process cancel button pressed.")
    return "Cancelling batch process...", "Cancelling any active generation..."

def get_next_filename(extension, output_dir="outputs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if extension.startswith("_"):
        base_name = extension[1:].split(".")[0]
        file_ext = "." + extension.split(".")[-1]
        
        existing_files = [f for f in os.listdir(output_dir) if base_name in f and f.endswith(file_ext)]
        current_numbers = []
        for file in existing_files:
            try:
                match = re.search(r'(\d+)_' + re.escape(base_name), file)
                if match:
                    current_numbers.append(int(match.group(1)))
            except:
                pass
        
        if not current_numbers:
            next_number = 1
        else:
            next_number = max(current_numbers) + 1
        
        return os.path.join(output_dir, f"{next_number}_{base_name}{file_ext}")
    else:
        counter = 1
        while True:
            filename = os.path.join(output_dir, f"{counter}{extension}")
            if not os.path.exists(filename):
                return filename
            counter += 1

def open_outputs_folder():
    outputs_dir = os.path.abspath("outputs")
    if os.name == 'nt':
        os.startfile(outputs_dir)
    elif os.name == 'posix':
        subprocess.Popen(["xdg-open", outputs_dir])
    else:
        print("[CMD] Opening folder not supported on this OS.")
    return f"Opened {outputs_dir}"

model_manager = ModelManager(device="cpu")

def load_wan_pipeline(model_choice, torch_dtype_str, num_persistent, lora_path=None, lora_alpha=None):
    print(f"[CMD] Loading model: {model_choice} with torch dtype: {torch_dtype_str} and num_persistent_param_in_dit: {num_persistent}")
    device = "cuda"
    torch_dtype = torch.float8_e4m3fn if torch_dtype_str == "torch.float8_e4m3fn" else torch.bfloat16
    global model_manager
    model_manager = ModelManager(device="cpu")
    if model_choice == "1.3B":
        t5_path = get_common_file(os.path.join("models", "models_t5_umt5-xxl-enc-bf16.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-T2V-1.3B", "models_t5_umt5-xxl-enc-bf16.pth"))
        vae_path = get_common_file(os.path.join("models", "Wan2.1_VAE.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth"))
        model_manager.load_models(
            [
                os.path.join("models", "Wan-AI", "Wan2.1-T2V-1.3B", "diffusion_pytorch_model.safetensors"),
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )
    elif model_choice == "14B_text":
        t5_path = get_common_file(os.path.join("models", "models_t5_umt5-xxl-enc-bf16.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "models_t5_umt5-xxl-enc-bf16.pth"))
        vae_path = get_common_file(os.path.join("models", "Wan2.1_VAE.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "Wan2.1_VAE.pth"))
        model_manager.load_models(
            [
                [
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00001-of-00006.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00002-of-00006.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00003-of-00006.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00004-of-00006.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00005-of-00006.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00006-of-00006.safetensors")
                ],
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )
    elif model_choice == "14B_image_720p":
        clip_path = get_common_file(os.path.join("models", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"))
        t5_path = get_common_file(os.path.join("models", "models_t5_umt5-xxl-enc-bf16.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "models_t5_umt5-xxl-enc-bf16.pth"))
        vae_path = get_common_file(os.path.join("models", "Wan2.1_VAE.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "Wan2.1_VAE.pth"))
        model_manager.load_models([clip_path], torch_dtype=torch.float32)
        model_manager.load_models(
            [
                [
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00001-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00002-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00003-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00004-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00005-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00006-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00007-of-00007.safetensors"),
                ],
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )        
    elif model_choice == "14B_image_480p":
        clip_path = get_common_file(os.path.join("models", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"))
        t5_path = get_common_file(os.path.join("models", "models_t5_umt5-xxl-enc-bf16.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "models_t5_umt5-xxl-enc-bf16.pth"))
        vae_path = get_common_file(os.path.join("models", "Wan2.1_VAE.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "Wan2.1_VAE.pth"))
        model_manager.load_models([clip_path], torch_dtype=torch.float32)
        model_manager.load_models(
            [
                [
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00001-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00002-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00003-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00004-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00005-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00006-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00007-of-00007.safetensors"),
                ],
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )
    else:
        raise ValueError("Invalid model choice")
    if lora_path is not None:
        if isinstance(lora_path, list):
            for path, alpha in lora_path:
                try:
                    print(f"[CMD] Loading LoRA from {path} with alpha {alpha}")
                    if os.path.exists(path):
                        model_manager.load_lora(path, lora_alpha=alpha)
                    else:
                        print(f"[CMD] Warning: LoRA file not found: {path}")
                except Exception as e:
                    print(f"[CMD] Error loading LoRA {path}: {e}")
        else:
            try:
                print(f"[CMD] Loading LoRA from {lora_path} with alpha {lora_alpha}")
                if os.path.exists(lora_path):
                    model_manager.load_lora(lora_path, lora_alpha=lora_alpha)
                else:
                    print(f"[CMD] Warning: LoRA file not found: {lora_path}")
            except Exception as e:
                print(f"[CMD] Error loading LoRA {lora_path}: {e}")
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    try:
        num_persistent_val = int(num_persistent)
    except:
        print("[CMD] Warning: could not parse num_persistent value, defaulting to 6000000000")
        num_persistent_val = 6000000000
    print(f"num_persistent_val {num_persistent_val}")
    pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_val)
    print("[CMD] Model loaded successfully.")
    return pipe

def get_lora_choices():
    lora_folder = "LoRAs"
    if not os.path.exists(lora_folder):
        os.makedirs(lora_folder)
        print("[CMD] 'LoRAs' folder not found. Created 'LoRAs' folder. Please add your LoRA .safetensors files.")
    files = [f for f in os.listdir(lora_folder) if f.endswith(".safetensors")]
    choices = ["None"] + sorted(files)
    return choices

def refresh_lora_list():
    update_val = gr.update(choices=get_lora_choices(), value="None")
    return update_val, update_val, update_val, update_val

def apply_fast_preset():
    return 20, True, 0.15, 5.6

# ------------------------- Main application with Gradio -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"],
                        help="The prompt extend method to use.")
    parser.add_argument("--prompt_extend_model", type=str, default=None, help="The prompt extend model to use.")
    parser.add_argument("--share", action="store_true", help="Share the Gradio app publicly.")
    args = parser.parse_args()
    loaded_pipeline = None
    loaded_pipeline_config = {}
    cancel_flag = False
    cancel_batch_flag = False
    prompt_expander = None
    with gr.Blocks() as demo:
        gr.Markdown("匹夫 Wan 2.1 I2V - V2V - T2V  | 匹夫的网站 :https://whli.ai | ")
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    generate_button = gr.Button("生成", variant="primary")
                    cancel_button = gr.Button("取消")
                    fast_preset_button = gr.Button("应用预设", variant="huggingface")
                    enhance_button = gr.Button("提示词增强", variant="primary")
                prompt_box = gr.Textbox(label="提示 (A <random: green , yellow , etc > car) 将随机抽取带有 ...等修饰的单词： 一辆黄色的汽车", placeholder="在此输入提示词", lines=5, value=config_loaded.get("prompt", ""))                
                with gr.Row():
                    gr.Markdown("### 模型 & 分辨率")
                with gr.Row():
                    model_choice_radio = gr.Radio(
                        choices=[
                            "WAN 2.1 1.3B (Text/Video-to-Video)",
                            "WAN 2.1 14B Text-to-Video",
                            "WAN 2.1 14B Image-to-Video 720P",
                            "WAN 2.1 14B Image-to-Video 480P"
                        ],
                        label="选择模型",
                        value=config_loaded.get("model_choice", "WAN 2.1 1.3B (Text/Video-to-Video)")
                    )
                    vram_preset_radio = gr.Radio(
                        choices=["4GB", "6GB", "8GB", "10GB", "12GB", "16GB", "24GB", "32GB", "48GB", "80GB"],
                        label="GPU VRAM预设",
                        value=config_loaded.get("vram_preset", "24GB")
                    )
                with gr.Row():
                    aspect_ratio_radio = gr.Radio(
                        # Instead of always using ASPECT_RATIOS_1_3b, determine which set to use based on model choice
                        choices=list(ASPECT_RATIOS_14b.keys() if 
                                    config_loaded.get("model_choice") in ["WAN 2.1 14B Text-to-Video", "WAN 2.1 14B Image-to-Video 720P"] 
                                    else ASPECT_RATIOS_1_3b.keys()),
                        label="宽高比例",
                        value=config_loaded.get("aspect_ratio", "16:9")
                    )
                with gr.Row():
                    extend_slider = gr.Slider(minimum=1, maximum=10, step=1, value=config_loaded.get("extend_factor", 1), label="扩展视频系数（1× = 无扩展）")
                    extension_info_button = gr.Button("扩展功能信息")
                with gr.Row():
                    extension_info_output = gr.Markdown("")
                with gr.Row():
                    num_generations = gr.Number(label="生成视频数量 ", value=config_loaded.get("num_generations", 1), precision=0)
                    width_slider = gr.Slider(minimum=320, maximum=1536, step=16, value=config_loaded.get("width", 832), label="宽")
                    height_slider = gr.Slider(minimum=320, maximum=1536, step=16, value=config_loaded.get("height", 480), label="高")
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            auto_crop_checkbox = gr.Checkbox(label="自动裁剪", value=config_loaded.get("auto_crop", True))
                            auto_scale_checkbox = gr.Checkbox(label="自动缩放", value=config_loaded.get("auto_scale", False))
                    with gr.Column():
                        tiled_checkbox = gr.Checkbox(label="平铺 VAE 解码（对于 12GB 或更多 GPU，1.3B 模型禁用）", value=config_loaded.get("tiled", True))
                    with gr.Column():
                        inference_steps_slider = gr.Slider(minimum=1, maximum=100, step=1, value=config_loaded.get("inference_steps", 50), label="生成步数")
                with gr.Row():
                    quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=config_loaded.get("quality", 5), label="质量")
                    fps_slider = gr.Slider(minimum=8, maximum=30, step=1, value=config_loaded.get("fps", 16), label="FPS (FPS（用于保存视频-您可以保存为8 FPS和4倍RIFE以获得2倍的持续时间）")
                    num_frames_slider = gr.Slider(minimum=1, maximum=300, step=1, value=config_loaded.get("num_frames", 81), label="帧数（始终为4x+1，例如17帧=1秒）。更多帧使用更多VRAM，速度较慢")
                gr.Markdown("### 使用Practical-RIFE提高视频FPS")
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            pr_rife_checkbox = gr.Checkbox(label="应用Practical-RIFE", value=config_loaded.get("pr_rife", True))
                            pr_rife_radio = gr.Radio(choices=["2x FPS", "4x FPS"], label="FPS乘数", value=config_loaded.get("pr_rife_multiplier", "2x FPS"))
                    with gr.Column():
                        with gr.Row():
                            cfg_scale_slider = gr.Slider(minimum=3, maximum=12, step=0.1, value=config_loaded.get("cfg_scale", 6.0), label="CFG 比例")
                            sigma_shift_slider = gr.Slider(minimum=3, maximum=12, step=0.1, value=config_loaded.get("sigma_shift", 6.0), label="Sigma 转移")
                gr.Markdown("### GPU 设置 - 如果出现 VRAM 错误或使用共享 VRAM，请减少此数字，FP8 可能会在 I2V 中产生颜色损坏")
                with gr.Row():
                    num_persistent_text = gr.Textbox(label="Dit 中的持久参数数量 (VRAM)", value=config_loaded.get("num_persistent", "12000000000"))
                    torch_dtype_radio = gr.Radio(
                        choices=["torch.float8_e4m3fn", "torch.bfloat16"],
                        label="FP8，在质量损失很小的情况下大大减少了显存使用。BF16（最高质量）",
                        value=config_loaded.get("torch_dtype", "torch.bfloat16")
                    )
                gr.Markdown("### TeaCache设置-在生成过程中显著加快生成速度-值太大会降低质量并导致失真")
                with gr.Row():
                    enable_teacache_checkbox = gr.Checkbox(label="启用TeaCache（建议1.3b型号阈值为0.05，14b型号阈值为0.15）", value=config_loaded.get("enable_teacache", False))
                with gr.Row():
                    tea_cache_l1_thresh_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=config_loaded.get("tea_cache_l1_thresh", 0.15), label="Tea Cache L1 阈值")
                    tea_cache_model_id_textbox = gr.Textbox(label="Tea Cache 模型 ID", value=config_loaded.get("tea_cache_model_id", "Wan2.1-T2V-1.3B"), placeholder="Enter Tea Cache Model ID")
                with gr.Row():
                    lora_dropdown = gr.Dropdown(
                        label="LoRA 模型（将 lora文件放在“LoRAs”文件夹中)",
                        choices=get_lora_choices(),
                        value=config_loaded.get("lora_model", "None")
                    )
                    lora_alpha_slider = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=config_loaded.get("lora_alpha", 1.0), label="LoRA 量表")
                    refresh_lora_button = gr.Button("刷新 LoRAs")
                with gr.Row():
                    show_more_lora_button = gr.Button("加载更多 LoRAs")
                    more_lora_state = gr.State(False)
                with gr.Column(visible=False) as more_lora_container:
                    lora_dropdown_2 = gr.Dropdown(label="LoRA Model 2", choices=get_lora_choices(), value=config_loaded.get("lora_model_2", "None"))
                    lora_alpha_slider_2 = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=config_loaded.get("lora_alpha_2", 1.0), label="LoRA Scale 2")
                    lora_dropdown_3 = gr.Dropdown(label="LoRA Model 3", choices=get_lora_choices(), value=config_loaded.get("lora_model_3", "None"))
                    lora_alpha_slider_3 = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=config_loaded.get("lora_alpha_3", 1.0), label="LoRA Scale 3")
                    lora_dropdown_4 = gr.Dropdown(label="LoRA Model 4", choices=get_lora_choices(), value=config_loaded.get("lora_model_4", "None"))
                    lora_alpha_slider_4 = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=config_loaded.get("lora_alpha_4", 1.0), label="LoRA Scale 4")
                show_more_lora_button.click(fn=toggle_lora_visibility, inputs=[more_lora_state], outputs=[more_lora_container, more_lora_state, show_more_lora_button])
                with gr.Row():
                    gr.Markdown("提示增强的目标语言：") 
                    tar_lang = gr.Radio(choices=["CH", "EN"], container=False, label="提示增强的目标语言：", value=config_loaded.get("tar_lang", "EN"))
                negative_prompt = gr.Textbox(label="负面提示", value=config_loaded.get("negative_prompt", "Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down"), placeholder="Enter negative prompt", lines=2)
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            save_prompt_checkbox = gr.Checkbox(label="将prompt保存到文件", value=config_loaded.get("save_prompt", True))
                            multiline_checkbox = gr.Checkbox(label="多行提示（每行单独显示）", value=config_loaded.get("multiline", False))               
                    with gr.Column():
                        with gr.Row():  
                            use_random_seed_checkbox = gr.Checkbox(label="使用随机种子", value=config_loaded.get("use_random_seed", True))
                            seed_input = gr.Textbox(label="种子 (固定种子)", placeholder="Enter seed", value=config_loaded.get("seed", ""))
                gr.Markdown("### 使用左面板将图像上传到视频，右面板将视频上传到视频（1.3b型号）或扩展现有视频（480p和720p I2V型号）")
                with gr.Row():
                    denoising_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=config_loaded.get("denoising_strength", 0.7),
                                             label="去噪强度（仅适用于视频到视频）")
                with gr.Row():
                    image_input = gr.Image(type="pil", label="输入图像（用于图像到视频）", height=512)
                    video_input = gr.Video(label="输入视频（适用于视频到视频，仅适用于1.3B）或扩展现有视频（使用最后一帧，适用于图像到视频模型）", format="mp4", height=512)
                with gr.Row():
                    clear_cache_checkbox = gr.Checkbox(label="生成后从RAM和VRAM中清除模型-目前效果不佳", value=config_loaded.get("clear_cache_after_gen", DEFAULT_CLEAR_CACHE))

            with gr.Column(scale=3):
                video_output = gr.Video(label="生成的视频", height=720)
                gr.Markdown("### 配置管理")
                with gr.Row():
                    config_name_textbox = gr.Textbox(label="配置名称（用于保存）", placeholder="输入配置名称", value="")
                    save_config_button = gr.Button("保存配置")
                with gr.Row():
                    config_dropdown = gr.Dropdown(label="加载配置", choices=get_config_list(), value=last_config)
                    load_config_button = gr.Button("加载配置")
                with gr.Row():
                    config_status = gr.Textbox(label="配置状态", value="", interactive=False, lines=1)
                gr.Markdown("### 批量图像到视频处理")
                batch_folder_input = gr.Textbox(label="批处理输入文件夹", placeholder="Enter input folder path", value=config_loaded.get("batch_folder", "batch_inputs"))
                batch_output_folder_input = gr.Textbox(label="批处理输出文件夹", placeholder="Enter batch outputs folder path", value=config_loaded.get("batch_output_folder", "batch_outputs"))
                with gr.Row():
                    skip_overwrite_checkbox = gr.Checkbox(label="如果输出存在，则跳过覆盖", value=config_loaded.get("skip_overwrite", True))
                    save_prompt_batch_checkbox = gr.Checkbox(label="将提示保存到文件（批处理）", value=config_loaded.get("save_prompt_batch", True))
                with gr.Row():
                    batch_process_button = gr.Button("批量处理", variant="primary")
                    cancel_batch_process_button = gr.Button("取消批处理")
                batch_status_output = gr.Textbox(label="批处理状态日志", lines=10)
                status_output = gr.Textbox(label="状态日志", lines=20)
                last_seed_output = gr.Textbox(label="上次使用的种子", interactive=False)
                open_outputs_button = gr.Button("打开输出文件夹")
        model_choice_radio.change(
            fn=update_model_settings,
            inputs=[model_choice_radio, vram_preset_radio, torch_dtype_radio],
            outputs=[aspect_ratio_radio, width_slider, height_slider, num_persistent_text]
        )
        vram_preset_radio.change(
            fn=update_model_settings,
            inputs=[model_choice_radio, vram_preset_radio, torch_dtype_radio],
            outputs=[aspect_ratio_radio, width_slider, height_slider, num_persistent_text]
        )
        torch_dtype_radio.change(
            fn=update_model_settings,
            inputs=[model_choice_radio, vram_preset_radio, torch_dtype_radio],
            outputs=[aspect_ratio_radio, width_slider, height_slider, num_persistent_text]
        )
        aspect_ratio_radio.change(
            fn=update_width_height,
            inputs=[aspect_ratio_radio, model_choice_radio],
            outputs=[width_slider, height_slider]
        )
        model_choice_radio.change(
            fn=update_tea_cache_model_id,
            inputs=[model_choice_radio],
            outputs=[tea_cache_model_id_textbox]
        )
        enhance_button.click(fn=prompt_enc, inputs=[prompt_box, tar_lang], outputs=prompt_box)
        generate_button.click(
            fn=generate_videos,
            inputs=[
                prompt_box, tar_lang, negative_prompt, image_input, video_input, denoising_slider,
                num_generations, save_prompt_checkbox, multiline_checkbox, use_random_seed_checkbox, seed_input,
                quality_slider, fps_slider,
                model_choice_radio, vram_preset_radio, num_persistent_text, torch_dtype_radio,
                num_frames_slider,
                aspect_ratio_radio, width_slider, height_slider, auto_crop_checkbox, auto_scale_checkbox, tiled_checkbox,
                inference_steps_slider, pr_rife_checkbox, pr_rife_radio, cfg_scale_slider, sigma_shift_slider,
                enable_teacache_checkbox, tea_cache_l1_thresh_slider, tea_cache_model_id_textbox,
                lora_dropdown, lora_alpha_slider,
                lora_dropdown_2, lora_alpha_slider_2,
                lora_dropdown_3, lora_alpha_slider_3,
                lora_dropdown_4, lora_alpha_slider_4,
                clear_cache_checkbox,
                extend_slider
            ],
            outputs=[video_output, status_output, last_seed_output]
        )
        cancel_button.click(fn=cancel_generation, outputs=status_output)
        fast_preset_button.click(fn=apply_fast_preset, inputs=[], outputs=[inference_steps_slider, enable_teacache_checkbox , tea_cache_l1_thresh_slider, sigma_shift_slider])
        open_outputs_button.click(fn=open_outputs_folder, outputs=status_output)
        batch_process_button.click(
            fn=batch_process_videos,
            inputs=[
                prompt_box,
                batch_folder_input, 
                batch_output_folder_input, 
                skip_overwrite_checkbox, 
                tar_lang, 
                negative_prompt, 
                denoising_slider,
                use_random_seed_checkbox, 
                seed_input, 
                quality_slider, 
                fps_slider, 
                model_choice_radio, 
                vram_preset_radio, 
                num_persistent_text,
                torch_dtype_radio, 
                num_frames_slider, 
                inference_steps_slider,
                aspect_ratio_radio, 
                width_slider, 
                height_slider, 
                auto_crop_checkbox, 
                auto_scale_checkbox,
                tiled_checkbox,
                cfg_scale_slider,
                sigma_shift_slider,
                save_prompt_batch_checkbox,
                pr_rife_checkbox, 
                pr_rife_radio,
                lora_dropdown, lora_alpha_slider,
                lora_dropdown_2, lora_alpha_slider_2,
                lora_dropdown_3, lora_alpha_slider_3,
                lora_dropdown_4, lora_alpha_slider_4,
                enable_teacache_checkbox, 
                tea_cache_l1_thresh_slider, 
                tea_cache_model_id_textbox,
                clear_cache_checkbox,
                extend_slider,
                num_generations
            ],
            outputs=batch_status_output
        )
        cancel_batch_process_button.click(fn=cancel_batch_process, outputs=[batch_status_output, status_output])
        load_config_button.click(
            fn=load_config,
            inputs=[config_dropdown],
            outputs=[
                config_status, 
                model_choice_radio, vram_preset_radio, aspect_ratio_radio, width_slider, height_slider,
                auto_crop_checkbox, auto_scale_checkbox, tiled_checkbox, inference_steps_slider, pr_rife_checkbox, pr_rife_radio, cfg_scale_slider, sigma_shift_slider,
                num_persistent_text, torch_dtype_radio,
                lora_dropdown, lora_alpha_slider,
                lora_dropdown_2, lora_alpha_slider_2,
                lora_dropdown_3, lora_alpha_slider_3,
                lora_dropdown_4, lora_alpha_slider_4,
                clear_cache_checkbox,
                negative_prompt, save_prompt_checkbox, multiline_checkbox, num_generations, use_random_seed_checkbox, seed_input,
                quality_slider, fps_slider, num_frames_slider, denoising_slider, tar_lang,
                batch_folder_input, batch_output_folder_input, skip_overwrite_checkbox, save_prompt_batch_checkbox,
                enable_teacache_checkbox, tea_cache_l1_thresh_slider, tea_cache_model_id_textbox,
                extend_slider,
                prompt_box
            ]
        )
        save_config_button.click(
            fn=save_config,
            inputs=[
                config_name_textbox, model_choice_radio, vram_preset_radio, aspect_ratio_radio, width_slider, height_slider,
                auto_crop_checkbox, auto_scale_checkbox, tiled_checkbox, inference_steps_slider, pr_rife_checkbox, pr_rife_radio, cfg_scale_slider, sigma_shift_slider,
                num_persistent_text, torch_dtype_radio,
                lora_dropdown, lora_alpha_slider,
                lora_dropdown_2, lora_alpha_slider_2,
                lora_dropdown_3, lora_alpha_slider_3,
                lora_dropdown_4, lora_alpha_slider_4,
                clear_cache_checkbox,
                negative_prompt, save_prompt_checkbox, multiline_checkbox, num_generations, use_random_seed_checkbox, seed_input,
                quality_slider, fps_slider, num_frames_slider, denoising_slider, tar_lang,
                batch_folder_input, batch_output_folder_input, skip_overwrite_checkbox, save_prompt_batch_checkbox,
                enable_teacache_checkbox, tea_cache_l1_thresh_slider, tea_cache_model_id_textbox,
                extend_slider,
                prompt_box
            ],
            outputs=[
                config_status, 
                config_dropdown,
                model_choice_radio, vram_preset_radio, aspect_ratio_radio, width_slider, height_slider,
                auto_crop_checkbox, auto_scale_checkbox, tiled_checkbox, inference_steps_slider, pr_rife_checkbox, pr_rife_radio, cfg_scale_slider, sigma_shift_slider,
                num_persistent_text, torch_dtype_radio,
                lora_dropdown, lora_alpha_slider,
                lora_dropdown_2, lora_alpha_slider_2,
                lora_dropdown_3, lora_alpha_slider_3,
                lora_dropdown_4, lora_alpha_slider_4,
                clear_cache_checkbox,
                negative_prompt, save_prompt_checkbox, multiline_checkbox, num_generations, use_random_seed_checkbox, seed_input,
                quality_slider, fps_slider, num_frames_slider, denoising_slider, tar_lang,
                batch_folder_input, batch_output_folder_input, skip_overwrite_checkbox, save_prompt_batch_checkbox,
                enable_teacache_checkbox, tea_cache_l1_thresh_slider, tea_cache_model_id_textbox,
                extend_slider,
                prompt_box
            ]
        )
        image_input.change(
            fn=update_target_dimensions,
            inputs=[image_input, auto_scale_checkbox, width_slider, height_slider],
            outputs=[width_slider, height_slider]
        )
        auto_scale_checkbox.change(
            fn=update_target_dimensions,
            inputs=[image_input, auto_scale_checkbox, width_slider, height_slider],
            outputs=[width_slider, height_slider]
        )
        extension_info_button.click(fn=show_extension_info, inputs=[], outputs=[extension_info_output])
        refresh_lora_button.click(fn=refresh_lora_list, outputs=[lora_dropdown, lora_dropdown_2, lora_dropdown_3, lora_dropdown_4])
        demo.launch(share=args.share, inbrowser=True)