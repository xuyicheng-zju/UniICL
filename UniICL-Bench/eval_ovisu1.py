"""
Ovis-U1 UniICL-Bench Evaluation Script
===================================
Supports all tasks: Understanding + T2I + I2I + Editing
"""
import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Any, List, Optional, Union
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig

# Add project roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import evaluators
import utils.judge
from public_path_config import (
    DEFAULT_JUDGE_MODEL,
    TASK_DATA_REL_PATHS,
    normalize_task_name,
)


class OvisU1Inferencer:
    """
    Ovis-U1 Inferencer wrapper.
    Handles Understanding (Interleaved), Text-to-Image, and Image Editing.
    """

    def __init__(self, model_path: str, device: str = "cuda", max_model_len: int = 32768):
        self.device = device
        self.max_model_len = max_model_len
        print(f"Loading Ovis-U1 from {model_path}...")
        print(f"Max model length: {max_model_len}")

        # Fix aimv2 registration conflict by patching AutoConfig.register
        original_register = AutoConfig.register
        def patched_register(model_type, config, exist_ok=False):
            if model_type == "aimv2":
                exist_ok = True
            return original_register(model_type, config, exist_ok=exist_ok)
        AutoConfig.register = patched_register

        # Load config first to modify max lengths
        from transformers import AutoConfig as TransformersAutoConfig
        config = TransformersAutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Set larger context window
        if hasattr(config, 'multimodal_max_length'):
            config.multimodal_max_length = max_model_len
            print(f"Set multimodal_max_length to {max_model_len}")

        if hasattr(config, 'llm_config') and hasattr(config.llm_config, 'max_position_embeddings'):
            original_max_pos = config.llm_config.max_position_embeddings
            print(f"Original max_position_embeddings: {original_max_pos}")

        self.model, loading_info = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            output_loading_info=True,
            trust_remote_code=True
        )
        print(f'Loading info of Ovis-U1:\n{loading_info}')
        self.model = self.model.eval().to(self.device).to(torch.bfloat16)
        print("Ovis-U1 loaded successfully!")

    def load_blank_image(self, width, height):
        return Image.new("RGB", (width, height), (255, 255, 255)).convert('RGB')

    def build_inputs_understanding(self, prompt, images):
        """Prepare inputs for understanding tasks (multi-image)"""
        text_tokenizer = self.model.get_text_tokenizer()
        visual_tokenizer = self.model.get_visual_tokenizer()
        
        # Ovis expects images as a list and prompt with <image> placeholders or interleaved logic
        # Based on test_multi_img_to_txt.py, we call preprocess_inputs with list of images
        # The prompt should contain <image> tokens if they are not added automatically by preprocess_inputs logic
        # However, looking at test_multi_img_to_txt.py: 
        # prompt = '\n'.join([f'Image {i+1}: <image>' for i in range(len(images))]) + '\n' + prompt
        # We need to ensure the prompt structure matches what preprocess_inputs expects.
        
        multimodal_type = 'multiple_image' if len(images) > 1 else 'single_image'
        if not images:
            multimodal_type = 'text_only' # Assuming, or handle separately? Ovis usually needs image?
            # If no images, we might just pass empty list, but let's check.
            # Usually benchmark tasks have images.
            pass

        prompt, input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            prompt, 
            images, 
            generation_preface='',
            return_labels=False,
            propagate_exception=False,
            multimodal_type=multimodal_type,
            fix_sample_overall_length_navit=False
        )
        
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        
        if pixel_values is not None:
            pixel_values = torch.cat([
                pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16) 
                if pixel_values is not None else None
            ], dim=0)
            
        if grid_thws is not None:
            grid_thws = torch.cat([
                grid_thws.to(device=visual_tokenizer.device) 
                if grid_thws is not None else None
            ], dim=0)
            
        return input_ids, pixel_values, attention_mask, grid_thws

    def build_inputs_generation(self, prompt, pil_image, target_width, target_height):
        """Prepare inputs for generation/editing"""
        visual_tokenizer = self.model.get_visual_tokenizer()
        text_tokenizer = self.model.get_text_tokenizer()

        vae_pixel_values = None
        
        if pil_image is not None:
            target_size = (int(target_width), int(target_height))
            # visual_generator is part of the model for generation
            pil_image, vae_pixel_values, cond_img_ids = self.model.visual_generator.process_image_aspectratio(pil_image, target_size)
            cond_img_ids[..., 0] = 1.0
            vae_pixel_values = vae_pixel_values.unsqueeze(0).to(device=self.model.device)
            width = pil_image.width
            height = pil_image.height
            resized_height, resized_width = visual_tokenizer.smart_resize(height, width, max_pixels=visual_tokenizer.image_processor.min_pixels)
            pil_image = pil_image.resize((resized_width, resized_height))

        # preprocess_inputs expects list of images
        images_input = [pil_image] if pil_image else []
        
        prompt, input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            prompt, 
            images_input, 
            generation_preface=None, # None for generation?
            return_labels=False,
            propagate_exception=False,
            multimodal_type='single_image', # Generation is usually single output?
            fix_sample_overall_length_navit=False
        )
        
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        
        if pixel_values is not None:
            pixel_values = torch.cat([
                pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16) if pixel_values is not None else None
            ], dim=0)
        if grid_thws is not None:
            grid_thws = torch.cat([
                grid_thws.to(device=visual_tokenizer.device) if grid_thws is not None else None
            ], dim=0)
            
        return input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values

    def interleave_inference(
        self,
        input_lists: List[Any],
        understanding_output: bool = True,
        use_editing_decoder: bool = False, # Not strictly used as flag, inferred from task
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        cfg_scale: float = 5.0, # txt_cfg default
        embedded_guidance: float = 3.5, # Not used in Ovis directly?
        seed: int = 42,
        **kwargs
    ) -> List[Any]:
        
        text_parts = []
        images = []
        
        # Parse inputs
        for item in input_lists:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, Image.Image):
                images.append(item)
            elif hasattr(item, 'convert'):
                images.append(item.convert("RGB"))

        # Construct Prompt
        # For understanding:
        # We need to construct a prompt that fits Ovis-U1's expectation.
        # UniICL-Bench usually gives: [img1, "Desc img1", img2, "Desc img2", ...]
        # Ovis needs explicit <image> tokens.
        
        if understanding_output:
            # Reconstruct the interleaved prompt with <image> tokens
            full_prompt = ""
            current_img_idx = 0
            
            for item in input_lists:
                if isinstance(item, str):
                    full_prompt += item
                else:
                    full_prompt += "<image>" 
            
            # Additional parameters for generate
            gen_kwargs = dict(
                max_new_tokens=2048,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=None,
                use_cache=True,
                eos_token_id=self.model.get_text_tokenizer().eos_token_id,
                pad_token_id=self.model.get_text_tokenizer().pad_token_id,
            )
            
            if "max_new_tokens" in kwargs:
                gen_kwargs["max_new_tokens"] = kwargs["max_new_tokens"]

            input_ids, pixel_values, attention_mask, grid_thws = self.build_inputs_understanding(full_prompt, images)
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids, 
                    pixel_values=pixel_values, 
                    attention_mask=attention_mask, 
                    grid_thws=grid_thws, 
                    **gen_kwargs
                )[0]
                text_tokenizer = self.model.get_text_tokenizer()
                gen_text = text_tokenizer.decode(output_ids, skip_special_tokens=True)
                
            return [gen_text]
        
        else:
            # Generation / Editing
            # If images is not empty, it's likely editing (using first image as reference)
            # If images is empty, it's T2I
            
            # Common params
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                use_cache=True,
                height=height,
                width=width,
                num_steps=num_inference_steps,
                seed=seed,
                eos_token_id=self.model.get_text_tokenizer().eos_token_id,
                pad_token_id=self.model.get_text_tokenizer().pad_token_id,
            )
            
            prompt_str = " ".join(text_parts)
            
            if images:
                # Image Editing
                input_img = images[0] # Use the first image as reference
                
                # Resize for internal processing if needed (similar to test_img_edit.py)
                # But here we stick to target width/height
                
                # Pipeline from test_img_edit.py
                # 1. Uncond generation (blank image + "Generate an image.")
                uncond_image = self.load_blank_image(width, height)
                uncond_prompt = "<image>\nGenerate an image."
                
                # img_cfg specific
                img_cfg = kwargs.get("img_cfg", 1.5)
                txt_cfg = cfg_scale
                
                gen_kwargs["img_cfg"] = img_cfg
                gen_kwargs["txt_cfg"] = txt_cfg
                
                visual_tokenizer = self.model.get_visual_tokenizer()
                text_tokenizer = self.model.get_text_tokenizer()
                
                input_ids, pixel_values, attention_mask, grid_thws, _ = self.build_inputs_generation(
                    uncond_prompt, uncond_image, width, height
                )
                
                with torch.inference_mode():
                    no_both_cond = self.model.generate_condition(
                        input_ids, pixel_values=pixel_values, 
                        attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs
                    )
                
                # 2. No-text condition (input image + "Generate an image.")
                # Resize input image to target
                input_img_resized = input_img.resize((width, height))
                with torch.inference_mode():
                    input_ids, pixel_values, attention_mask, grid_thws, _ = self.build_inputs_generation(
                        uncond_prompt, input_img_resized, width, height
                    )
                    no_txt_cond = self.model.generate_condition(
                        input_ids, pixel_values=pixel_values, 
                        attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs
                    )

                # 3. Full condition (input image + prompt)
                full_prompt = "<image>\n" + prompt_str.strip()
                input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values = self.build_inputs_generation(
                    full_prompt, input_img_resized, width, height
                )
                
                with torch.inference_mode():
                    cond = self.model.generate_condition(
                        input_ids, pixel_values=pixel_values, 
                        attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs
                    )
                    cond["vae_pixel_values"] = vae_pixel_values
                    res_images = self.model.generate_img(
                        cond=cond, no_both_cond=no_both_cond, 
                        no_txt_cond=no_txt_cond, **gen_kwargs
                    )
                return res_images
                
            else:
                # Text-to-Image
                # Pipeline from test_txt_to_img.py
                
                gen_kwargs["img_cfg"] = 0
                gen_kwargs["txt_cfg"] = cfg_scale
                
                uncond_image = self.load_blank_image(width, height)
                uncond_prompt = "<image>\nGenerate an image."
                
                input_ids, pixel_values, attention_mask, grid_thws, _ = self.build_inputs_generation(
                    uncond_prompt, uncond_image, width, height
                )
                
                with torch.inference_mode():
                    no_both_cond = self.model.generate_condition(
                        input_ids, pixel_values=pixel_values, 
                        attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs
                    )

                # Prompt formatting from T2I test: 
                # "<image>\nDescribe the image by detailing ... :" + prompt
                # But for benchmark, we usually just want the user prompt. 
                # Let's stick to simple prompt unless the model STRICTLY requires the prefix.
                # test_txt_to_img.py adds a long prefix. This might be "God Prompt".
                # For fairness in benchmark, we usually use the raw prompt. 
                # However, Ovis might be tuned with this. 
                # Let's use the provided prompt directly + <image> prefix for now.
                
                full_prompt = "<image>\n" + prompt_str
                # Note: test script does: "<image>\nDescribe...:" + prompt
                # If quality is bad, we might need to add that back.
                
                no_txt_cond = None
                input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values = self.build_inputs_generation(
                    full_prompt, uncond_image, width, height
                )
                
                with torch.inference_mode():
                    cond = self.model.generate_condition(
                        input_ids, pixel_values=pixel_values, 
                        attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs
                    )
                    cond["vae_pixel_values"] = vae_pixel_values # Uncond image VAE? Wait, test script uses uncond_image for T2I input
                    # YES: build_inputs(..., uncond_image, ...)
                    
                    res_images = self.model.generate_img(
                        cond=cond, no_both_cond=no_both_cond, 
                        no_txt_cond=no_txt_cond, **gen_kwargs
                    )
                return res_images


def main():
    parser = argparse.ArgumentParser(description="Ovis-U1 UniICL-Bench Evaluation")
    parser.add_argument("--model-path", type=str, default="AIDC-AI/Ovis-U1-3B", help="Path to Ovis-U1 model")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--max-model-len", type=int, default=32768, help="Maximum model context length (default: 32768)")
    parser.add_argument("--judge-api-base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--image-dir", type=str)
    parser.add_argument("--degraded-dir", type=str, default=None,
                       help="Directory for degraded images (visual_refinement task)")
    parser.add_argument("--gt-dir", type=str, default=None,
                       help="Directory for GT images (visual_refinement task)")
    parser.add_argument("--benchmark-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="./eval_results_ovisu1")
    parser.add_argument("--k-shot", type=int, default=0)
    parser.add_argument("--hps-checkpoint", type=str, default=None)
    args = parser.parse_args()
    try:
        args.task = normalize_task_name(args.task, allow_all=True)
    except ValueError as e:
        raise SystemExit(str(e))

    # Judge Config
    utils.judge.VLLM_API_BASE = args.judge_api_base
    utils.judge.JUDGE_MODEL = args.judge_model

    TASK_DATA_MAP = TASK_DATA_REL_PATHS
    
    # (function, model_type)
    # model_type: "hps" (HPSv3), "qalign" (Q-Align), None
    TASK_CONFIG = {
        "visual_grounding": (evaluators.eval_grounding, None),
        "attribute_recognition": (evaluators.eval_attr_rec_gen, None),
        "scene_reasoning": (evaluators.eval_vqa_gen, None),
        "style_aware_caption": (evaluators.eval_caption_styled, None),
        "instructional_generation": (evaluators.eval_t2i, "hps"),
        "image_manipulation": (evaluators.eval_i2i_editing, None),
        "aesthetic_assessment": (evaluators.eval_aesthetic_assessment, None),
        "forgery_detection": (evaluators.eval_authenticity_detection, None),
        "visual_refinement": (evaluators.eval_image_perfection, "qalign"),
        "fast_concept_mapping": (evaluators.eval_fcb_classification, None),
        "fast_concept_generation": (evaluators.eval_fci_t2i, None),
        "world_aware_planning": (evaluators.eval_planning, None),
        "chain_of_editing": (evaluators.eval_chain_of_editing, None),
        "analogical_editing": (evaluators.eval_visualcloze_g, None),
        "analogical_inference": (evaluators.eval_visualcloze_u, None),
    }

    if args.task and args.task != "all" and not args.data_path:
        if args.task in TASK_DATA_MAP:
            args.data_path = os.path.join(args.benchmark_dir, TASK_DATA_MAP[args.task])

    if not args.image_dir:
        print("Error: --image-dir is required")
        return


    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)


    inferencer = OvisU1Inferencer(args.model_path, args.device, max_model_len=args.max_model_len)


    hps_model = None
    qalign_model = None


    if args.task in ["instructional_generation", "all"]:
        if args.hps_checkpoint and os.path.exists(args.hps_checkpoint):
            print("\nLoading HPSv3 model for Instructional Generation evaluation...")
            from utils.scoring import load_hpsv3_model
            hps_model = load_hpsv3_model(args.hps_checkpoint)
            if hps_model:
                print("✅ HPSv3 model loaded successfully")
            else:
                print("⚠️ HPSv3 model not loaded, Instructional Generation scoring will be skipped")


    if args.task in ["visual_refinement", "all"]:
        print("\nLoading Q-Align model for Visual Refinement evaluation...")
        from utils.scoring import load_qalign_model
        try:
            qalign_model = load_qalign_model()
            print("✅ Q-Align model loaded successfully")
        except Exception as e:
            print(f"⚠️ Q-Align model not loaded: {e}")

    def run_task(task_name, data_path, image_dir):
        if task_name not in TASK_CONFIG:
            print(f"Skip {task_name}")
            return

        eval_func, model_type = TASK_CONFIG[task_name]
        output_path = os.path.join(output_dir, f"{task_name}_results.json")


        if task_name == "visual_refinement":
            eval_func(inferencer, data_path, image_dir, output_path, args.k_shot,
                     qalign_model=qalign_model, degraded_dir=args.degraded_dir, gt_dir=args.gt_dir)
        elif task_name in ["chain_of_editing"]:
            eval_func(inferencer, data_path, image_dir, output_path)
        elif model_type == "hps":
            eval_func(inferencer, data_path, image_dir, output_path, args.k_shot, hps_model=hps_model)
        else:
            eval_func(inferencer, data_path, image_dir, output_path, args.k_shot)

    if args.task == "all":
        print("Use run_eval.sh for task=all")
    else:
        run_task(args.task, args.data_path, args.image_dir)

if __name__ == "__main__":
    main()
