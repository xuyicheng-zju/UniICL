"""Public release module documentation."""
import os
import sys
import json
import argparse
import torch
import numpy as np
from public_path_config import DEFAULT_FLUX_MODEL, DEFAULT_JUDGE_MODEL, DEFAULT_NEXUSGEN_MODEL, TASK_DATA_REL_PATHS, normalize_task_name
from pathlib import Path
from PIL import Image
from typing import Any, List, Optional, Union
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


NEXUSGEN_ROOT = PROJECT_ROOT / "Nexus-Gen"
sys.path.insert(0, str(NEXUSGEN_ROOT))

from utils import evaluators
import utils.judge


class NexusGenInferencer:
    """Public release documentation."""

    def __init__(
        self,
        model_path: str,
        flux_path: str,
        device: str = "cuda:0",
        enable_cpu_offload: bool = False,
        fp8_quantization: bool = False,
        max_pixels: int = 262640,
    ):
        self.device = device
        self.enable_cpu_offload = enable_cpu_offload
        self.fp8_quantization = fp8_quantization
        self.max_pixels = max_pixels
        self.model_path = model_path
        self.flux_path = flux_path

        print(f"Loading Nexus-Gen V2 from {model_path}...")

        from transformers import AutoConfig
        from qwen_vl_utils import smart_resize
        from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor

        self.smart_resize = smart_resize


        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            config=model_config,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device
        )
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        self.model.eval()


        self.generation_decoder_path = os.path.join(model_path, "generation_decoder.bin")
        self.editing_decoder_path = os.path.join(model_path, "edit_decoder.bin")


        self._generation_decoder = None
        self._editing_decoder = None

        print("Nexus-Gen V2 loaded successfully!")

    def _get_generation_decoder(self):
        """Public release documentation."""
        if self._generation_decoder is None:
            from modeling.decoder.generation_decoder import NexusGenGenerationDecoder
            self._generation_decoder = NexusGenGenerationDecoder(
                self.generation_decoder_path,
                self.flux_path,
                device=self.device,
                enable_cpu_offload=self.enable_cpu_offload,
                fp8_quantization=self.fp8_quantization
            )
        return self._generation_decoder

    def _get_editing_decoder(self):
        """Public release documentation."""
        if self._editing_decoder is None:
            from modeling.decoder.editing_decoder import NexusGenEditingDecoder
            self._editing_decoder = NexusGenEditingDecoder(
                self.editing_decoder_path,
                self.flux_path,
                self.model_path,
                device=self.device,
                enable_cpu_offload=self.enable_cpu_offload,
                fp8_quantization=self.fp8_quantization
            )
        return self._editing_decoder

    def _bound_image(self, image: Image.Image) -> Image.Image:
        """Public release documentation."""
        resized_height, resized_width = self.smart_resize(
            image.height,
            image.width,
            factor=28,  # Qwen2.5-VL vision encoder patch size
            max_pixels=self.max_pixels,
        )
        return image.resize((resized_width, resized_height))

    def get_processed_image_size(self, image: Image.Image) -> tuple:
        """Public release documentation."""



        

        resized_height, resized_width = self.smart_resize(
            image.height,
            image.width,
            factor=28,
            max_pixels=self.max_pixels,
        )
        


        return (resized_width, resized_height)
    
    def get_bbox_normalization_size(self, image: Image.Image, bbox_raw: list = None) -> tuple:
        """Public release documentation."""

        orig_w, orig_h = image.size
        proc_w, proc_h = self.get_processed_image_size(image)
        
        if bbox_raw is None:

            return (proc_w, proc_h, "processed")
        
        x1, y1, x2, y2 = bbox_raw
        max_coord = max(x1, y1, x2, y2)
        


        if x2 <= proc_w * 1.1 and y2 <= proc_h * 1.1:  # 10% tolerance
            return (proc_w, proc_h, "processed")
        

        if max_coord <= 1000 and max_coord > 100:
            return (1000, 1000, "normalized_1000")
        

        if x2 <= orig_w * 1.1 and y2 <= orig_h * 1.1:
            return (orig_w, orig_h, "original")
        

        return (proc_w, proc_h, "processed_fallback")

    def _get_image_embedding_for_edit(self, image: Image.Image, target_size=(504, 504)):
        """Public release documentation."""
        image = image.resize(target_size, Image.BILINEAR)
        inputs = self.processor.image_processor(
            images=[image], videos=None, return_tensors='pt', do_resize=False
        )
        pixel_values = inputs["pixel_values"].to(self.model.device)
        image_grid_thw = inputs["image_grid_thw"].to(self.model.device)
        pixel_values = pixel_values.type(self.model.visual.dtype)
        with torch.no_grad():
            image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def interleave_inference(
        self,
        input_lists: List[Any],
        understanding_output: bool = True,
        use_editing_decoder: bool = False,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        cfg_scale: float = 3.0,
        embedded_guidance: float = 3.5,
        seed: int = 42,
        **kwargs
    ) -> List[Any]:
        """Public release documentation."""




        images = []
        text_parts = []
        last_image_for_edit = None

        IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

        for item in input_lists:
            if isinstance(item, str):


                text_parts.append(item)
            elif isinstance(item, Image.Image):
                images.append(item)
                last_image_for_edit = item
                text_parts.append(IMAGE_PLACEHOLDER)
            elif hasattr(item, 'convert'):
                img = item.convert("RGB")
                images.append(img)
                last_image_for_edit = img
                text_parts.append(IMAGE_PLACEHOLDER)


        full_text = "".join(text_parts)


        messages = [{"role": "user", "content": [{"type": "text", "text": full_text}]}]


        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


        if understanding_output:
            print(f"\n[DEBUG Analogical Inference Check]")
            print(f"[DEBUG] Number of images: {len(images)}")
            print(f"[DEBUG] full_text length: {len(full_text)}")
            print(f"[DEBUG] full_text preview: {full_text[:500]}...")
            print(f"[DEBUG] Generated text preview: {text[:500]}...")
            print(f"[DEBUG] Image placeholders in full_text: {full_text.count(IMAGE_PLACEHOLDER)}")
            print(f"[DEBUG] ===\n")


        if images:
            image_inputs = [self._bound_image(img) for img in images]
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )

        inputs = inputs.to(self.model.device)

        if understanding_output:

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=2048, temperature=0.1, do_sample=False)

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            return output_text

        else:

            generation_image_grid_thw = torch.tensor([[1, 18, 18]]).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    return_dict_in_generate=True,
                    generation_image_grid_thw=generation_image_grid_thw
                )


            if not hasattr(outputs, 'output_image_embeddings') or outputs.output_image_embeddings is None:
                print("Warning: No image embeddings generated")
                return [None]

            output_image_embeddings = outputs.output_image_embeddings


            if self.enable_cpu_offload:
                self.model.cpu()
                torch.cuda.empty_cache()


            if use_editing_decoder and last_image_for_edit is not None:

                ref_embeddings = self._get_image_embedding_for_edit(last_image_for_edit)
                decoder = self._get_editing_decoder()
                image = decoder.decode_image_embeds(
                    output_image_embeddings,
                    ref_embed=ref_embeddings,
                    height=height,
                    width=width,
                    negative_prompt="",
                    cfg_scale=1.0,
                    num_inference_steps=num_inference_steps,
                    embedded_guidance=embedded_guidance,
                    seed=seed
                )
            else:

                decoder = self._get_generation_decoder()
                image = decoder.decode_image_embeds(
                    output_image_embeddings,
                    height=height,
                    width=width,
                    negative_prompt="",
                    cfg_scale=cfg_scale,
                    num_inference_steps=num_inference_steps,
                    embedded_guidance=embedded_guidance,
                    seed=seed
                )


            if self.enable_cpu_offload:
                self.model.to(self.device)

            return [image]


def parse_bbox(text):
    """Public release documentation."""
    import re
    pattern = r'\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?'
    match = re.search(pattern, text)
    if match:
        try:
            coords = [float(match.group(i)) for i in range(1, 5)]
            return coords
        except ValueError:
            return None
    return None


def compute_iou(box1, box2):
    """Public release documentation."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def normalize_bbox_adaptive(bbox_raw, img_width, img_height, proc_width, proc_height):
    """Public release documentation."""
    if bbox_raw is None:
        return None, "none"
    
    x1, y1, x2, y2 = bbox_raw
    max_coord = max(x1, y1, x2, y2)
    

    if max_coord <= 1.0:
        return [x1, y1, x2, y2], "already_normalized"
    

    if x2 <= proc_width * 1.1 and y2 <= proc_height * 1.1:
        norm_bbox = [
            min(x1 / proc_width, 1.0),
            min(y1 / proc_height, 1.0),
            min(x2 / proc_width, 1.0),
            min(y2 / proc_height, 1.0)
        ]
        return norm_bbox, "processed"
    

    if max_coord <= 1000:
        norm_bbox = [
            min(x1 / 1000.0, 1.0),
            min(y1 / 1000.0, 1.0),
            min(x2 / 1000.0, 1.0),
            min(y2 / 1000.0, 1.0)
        ]
        return norm_bbox, "normalized_1000"
    

    if x2 <= img_width * 1.1 and y2 <= img_height * 1.1:
        norm_bbox = [
            min(x1 / img_width, 1.0),
            min(y1 / img_height, 1.0),
            min(x2 / img_width, 1.0),
            min(y2 / img_height, 1.0)
        ]
        return norm_bbox, "original"
    

    norm_bbox = [
        min(x1 / proc_width, 1.0),
        min(y1 / proc_height, 1.0),
        min(x2 / proc_width, 1.0),
        min(y2 / proc_height, 1.0)
    ]
    return norm_bbox, "fallback"


def eval_grounding_nexusgen(inferencer, data_path, image_dir, output_path, num_demos=3):
    """Public release documentation."""
    from utils.icl import build_icl_input
    
    print(f"\n=== Evaluating Visual Grounding for NexusGen ({num_demos}-shot) ===")

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    results = []
    total_iou = 0.0
    valid_count = 0
    strategy_counts = {}

    inference_params = dict(
        max_think_token_n=2048,
    )

    for item in tqdm(data, desc=f"Visual Grounding {num_demos}shot"):
        image_path = os.path.join(image_dir, item['image_name'])
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            target_img = Image.open(image_path).convert("RGB")
            img_width, img_height = target_img.size
        except Exception as e:
            print(f"Cannot read image {image_path}: {e}")
            continue


        if hasattr(inferencer, 'get_processed_image_size'):
            proc_width, proc_height = inferencer.get_processed_image_size(target_img)
        else:
            proc_width, proc_height = img_width, img_height

        demos = item['demos'][:num_demos] if num_demos > 0 else []

        question = item.get('instruction', item.get('text', ''))
        input_list = build_icl_input(demos, image_dir, image_path, question)

        try:
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=True,
                think=False,
                **inference_params
            )
            prediction = output_list[-1].strip()
        except Exception as e:
            print(f"Error processing {item['image_name']}: {e}")
            prediction = ""


        pred_bbox_raw = parse_bbox(prediction)
        

        pred_bbox, norm_strategy = normalize_bbox_adaptive(
            pred_bbox_raw, img_width, img_height, proc_width, proc_height
        )
        

        strategy_counts[norm_strategy] = strategy_counts.get(norm_strategy, 0) + 1


        gt_bbox_raw = item.get('answer', item.get('annotation', []))
        if isinstance(gt_bbox_raw, str):
            gt_bbox = parse_bbox(gt_bbox_raw)
        elif isinstance(gt_bbox_raw, list):
            gt_bbox = gt_bbox_raw
        else:
            gt_bbox = None


        iou = 0.0
        if pred_bbox is not None and gt_bbox is not None:
            iou = compute_iou(pred_bbox, gt_bbox)
            total_iou += iou
            valid_count += 1

        results.append({
            'image_name': item['image_name'],
            'question': item.get('instruction', item.get('text', '')),
            'ground_truth': gt_bbox,
            'prediction': prediction,
            'pred_bbox_raw': pred_bbox_raw,
            'pred_bbox_normalized': pred_bbox,
            'image_size': [img_width, img_height],
            'processed_size': [proc_width, proc_height],
            'norm_strategy': norm_strategy,
            'iou': iou
        })

    mean_iou = total_iou / valid_count if valid_count > 0 else 0.0
    print(f"\nMean IoU: {mean_iou:.4f} ({valid_count}/{len(data)} valid predictions)")
    print(f"Normalization strategy distribution: {strategy_counts}")

    result_data = {
        'mean_iou': mean_iou,
        'valid_count': valid_count,
        'total_count': len(data),
        'strategy_distribution': strategy_counts,
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Nexus-Gen V2 UniICL-Bench Evaluation")
    parser.add_argument("--model-path", type=str, default=DEFAULT_NEXUSGEN_MODEL,
                        help="Path to Nexus-GenV2 model")
    parser.add_argument("--flux-path", type=str, default=DEFAULT_FLUX_MODEL,
                        help="Path to FLUX models directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--judge-api-base", type=str, default="http://localhost:8000/v1",
                        help="vLLM API base URL for judge model")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help="Judge model name")
    parser.add_argument("--task", type=str, default="all",
                        help="Paper-aligned task name in snake_case, or all")
    parser.add_argument("--data-path", type=str, help="Task data path")
    parser.add_argument("--image-dir", type=str, help="Image root directory")
    parser.add_argument("--benchmark-dir", type=str, default=".", help="UniICL-Bench root directory")
    parser.add_argument("--output-dir", type=str, default="./eval_results_nexusgen", help="Output directory")
    parser.add_argument("--k-shot", type=int, default=0, help="ICL k-shot setting")
    parser.add_argument("--hps-checkpoint", type=str, default=None, help="HPSv3 checkpoint path")
    parser.add_argument("--enable-cpu-offload", action="store_true", default=True,
                        help="Enable CPU offloading for memory optimization")
    parser.add_argument("--fp8-quantization", action="store_true", default=False,
                        help="Enable FP8 quantization")
    args = parser.parse_args()
    try:
        args.task = normalize_task_name(args.task, allow_all=True)
    except ValueError as e:
        parser.error(str(e))


    utils.judge.VLLM_API_BASE = args.judge_api_base
    utils.judge.JUDGE_MODEL = args.judge_model


    TASK_DATA_MAP = TASK_DATA_REL_PATHS


    # model_type: "hps", "qalign", None
    TASK_CONFIG = {
        "visual_grounding": (evaluators.eval_grounding, None),
        "attribute_recognition": (evaluators.eval_attr_rec_gen, None),
        "scene_reasoning": (evaluators.eval_vqa_gen, None),
        "style_aware_caption": (evaluators.eval_caption_styled, None),
        "instructional_generation": (evaluators.eval_t2i, "hps"),
        "image_manipulation": (evaluators.eval_i2i_editing, "hps"),
        "aesthetic_assessment": (evaluators.eval_aesthetic_assessment, None),
        "forgery_detection": (evaluators.eval_authenticity_detection, None),
        "visual_refinement": (evaluators.eval_image_perfection, "hps"),
        "fast_concept_mapping": (evaluators.eval_fcb_classification, None),
        "fast_concept_generation": (evaluators.eval_fci_t2i, "hps"),
        "world_aware_planning": (evaluators.eval_planning, None),
        "chain_of_editing": (evaluators.eval_chain_of_editing, "hps"),
        "analogical_editing": (evaluators.eval_visualcloze_g, "hps"),
        "analogical_inference": (evaluators.eval_visualcloze_u, None),
    }

    if args.task and args.task != "all" and not args.data_path:
        if args.task in TASK_DATA_MAP:
            args.data_path = os.path.join(args.benchmark_dir, TASK_DATA_MAP[args.task])


    if not args.image_dir:
        parser.error("--image-dir is required. Use run_eval.sh for automatic path resolution.")

    if args.k_shot < 0:
        parser.error(f"--k-shot must be non-negative, got {args.k_shot}")


    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)


    inferencer = NexusGenInferencer(
        model_path=args.model_path,
        flux_path=args.flux_path,
        device=args.device,
        enable_cpu_offload=args.enable_cpu_offload,
        fp8_quantization=args.fp8_quantization,
    )


    hps_model = None
    if args.task == "instructional_generation" and args.hps_checkpoint and os.path.exists(args.hps_checkpoint):
        print(f"Loading HPSv3 model from {args.hps_checkpoint}...")
        hps_model = evaluators.load_hpsv3_model(args.hps_checkpoint)


    HPS_TASKS = {"instructional_generation"}

    def run_task(task_name: str, data_path: str, image_dir: str):
        if task_name not in TASK_CONFIG:
            print(f"Skipping {task_name}: unsupported")
            return

        eval_func, model_type = TASK_CONFIG[task_name]

        if not os.path.exists(data_path):
            print(f"Skipping {task_name}: data file not found at {data_path}")
            return

        output_path = os.path.join(output_dir, f"{task_name}_results.json")


        if task_name == "visual_refinement":

            eval_func(inferencer, data_path, image_dir, output_path, args.k_shot)
        elif task_name in HPS_TASKS and hps_model is not None:
            eval_func(inferencer, data_path, image_dir, output_path, args.k_shot, hps_model=hps_model)
        else:
            eval_func(inferencer, data_path, image_dir, output_path, args.k_shot)

    if args.task == "all":
        parser.error("--task all is not supported. Use run_eval.sh to run all tasks with correct image directories.")
    else:
        data_path = args.data_path or os.path.join(args.benchmark_dir, TASK_DATA_MAP.get(args.task, ""))
        run_task(args.task, data_path, args.image_dir)

    print(f"\nEvaluation completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
