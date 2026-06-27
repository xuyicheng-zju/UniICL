"""
Qwen3-VL UniICL-Bench Evaluation (Understanding-only) via Local vLLM
-----------------------------------------------------------------
Qwen3-VL uses [0,1000] normalized coordinates for grounding tasks.
Uses local vLLM for inference.
"""
import argparse
import json
import os
import re
import sys
import torch
from pathlib import Path
from typing import Any, List
from PIL import Image
from tqdm import tqdm

# Add project roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import evaluators
import utils.judge  # noqa: E402
from public_path_config import DEFAULT_JUDGE_MODEL, TASK_DATA_REL_PATHS, normalize_task_name


class Qwen3VLInferencer:
    """Local vLLM inference wrapper for Qwen3-VL."""

    def __init__(self, model_path: str, max_tokens: int = 2048, tensor_parallel_size: int = 1):
        from vllm import LLM, SamplingParams

        self.model_path = model_path
        self.max_tokens = max_tokens
        print(f"Loading Qwen3-VL from {model_path}...")

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=32768,
            limit_mm_per_prompt={"image": 20},
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=max_tokens,
        )
        print("Qwen3-VL loaded successfully!")

    def interleave_inference(self, input_lists: List[Any], understanding_output: bool, **kwargs):
        from vllm import SamplingParams

        if not understanding_output:
            raise ValueError("Qwen3-VL only supports understanding_output=True")

        # Build prompt and collect images
        prompt_parts = []
        images = []

        for term in input_lists:
            if isinstance(term, str):
                prompt_parts.append(term)
            else:
                # PIL.Image - add placeholder and collect image
                images.append(term)
                prompt_parts.append("<|vision_start|><|image_pad|><|vision_end|>")

        prompt = "".join(prompt_parts)

        # Build vLLM input
        if images:
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": images},
            }
        else:
            inputs = {"prompt": prompt}

        # Override sampling params if needed
        max_tokens = kwargs.get("max_think_token_n", self.max_tokens)
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=max_tokens,
        )

        try:
            outputs = self.llm.generate([inputs], sampling_params=sampling_params)
            content = outputs[0].outputs[0].text.strip()
        except Exception as e:
            print(f"vLLM inference failed: {e}")
            content = ""

        return [content]


def parse_bbox(text):
    """Public release documentation."""
    pattern = r'\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?'
    match = re.search(pattern, text)
    if match:
        try:
            coords = [float(match.group(i)) for i in range(1, 5)]
            return coords
        except ValueError:
            return None
    return None


def convert_bbox_to_qwen3_normalized(bbox_str):
    """Public release documentation."""
    bbox = parse_bbox(bbox_str)
    if bbox is None:
        return bbox_str

    x1, y1, x2, y2 = bbox
    norm_x1 = int(x1 * 1000)
    norm_y1 = int(y1 * 1000)
    norm_x2 = int(x2 * 1000)
    norm_y2 = int(y2 * 1000)

    return f"[{norm_x1}, {norm_y1}, {norm_x2}, {norm_y2}]"


def build_icl_input_qwen3_coords(demos, image_dir, target_image_path, target_question):
    """Public release documentation."""
    input_list = []

    for demo in demos:
        demo_img_path = os.path.join(image_dir, demo['image_name'])
        if os.path.exists(demo_img_path):
            try:
                demo_img = Image.open(demo_img_path).convert("RGB")
            except Exception:
                continue

            input_list.append(demo_img)

            demo_question = demo.get('instruction', demo.get('text', ''))
            demo_answer = demo.get('answer', demo.get('annotation', ''))
            demo_answer_qwen3 = convert_bbox_to_qwen3_normalized(demo_answer)

            demo_text = f"User: {demo_question} Assistant: {demo_answer_qwen3}"
            input_list.append(demo_text)

    target_img = Image.open(target_image_path).convert("RGB")
    input_list.append(target_img)
    input_list.append(f"User: {target_question} Assistant: ")

    return input_list


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


def eval_grounding_qwen3vl(inferencer, data_path, image_dir, output_path, num_demos=3):
    """Public release documentation."""
    print(f"\n=== Evaluating Visual Grounding for Qwen3-VL ({num_demos}-shot) ===")

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    results = []
    total_iou = 0.0
    valid_count = 0

    inference_params = dict(
        max_think_token_n=2048,
    )

    for item in tqdm(data, desc=f"Visual Grounding {num_demos}shot"):
        image_path = os.path.join(image_dir, item['image_name'])
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Cannot read image {image_path}: {e}")
            continue

        demos = item['demos'][:num_demos] if num_demos > 0 else []

        question = item.get('instruction', item.get('text', ''))
        input_list = build_icl_input_qwen3_coords(demos, image_dir, image_path, question)

        try:
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=True,
                **inference_params
            )
            prediction = output_list[-1].strip()
        except Exception as e:
            print(f"Error processing {item['image_name']}: {e}")
            prediction = ""


        pred_bbox_raw = parse_bbox(prediction)
        if pred_bbox_raw is not None:
            pred_bbox = [coord / 1000.0 for coord in pred_bbox_raw]
        else:
            pred_bbox = None

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
            'iou': iou
        })

    mean_iou = total_iou / valid_count if valid_count > 0 else 0.0
    print(f"\nMean IoU: {mean_iou:.4f} ({valid_count}/{len(data)} valid predictions)")

    result_data = {
        'mean_iou': mean_iou,
        'valid_count': valid_count,
        'total_count': len(data),
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL UniICL-Bench Evaluation via Local vLLM")
    parser.add_argument("--model-path", type=str, required=True, help="Path to Qwen3-VL model")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--judge-api-base", type=str, default="http://localhost:8000/v1", help="Judge API base URL")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL, help="Judge model name")
    parser.add_argument("--task", type=str, default="all",
                        help="Paper-aligned task name in snake_case, or all")
    parser.add_argument("--data-path", type=str, help="Task data path (auto when not set)")
    parser.add_argument("--image-dir", type=str, help="Image root (auto when not set)")
    parser.add_argument("--benchmark-dir", type=str, default=".", help="UniICL-Bench root directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (auto-generated from model name if not specified)")
    parser.add_argument("--k-shot", type=int, default=0, help="ICL k-shot setting")
    args = parser.parse_args()
    try:
        args.task = normalize_task_name(args.task, allow_all=True)
    except ValueError as e:
        parser.error(str(e))


    if args.output_dir is None:
        model_name_clean = os.path.basename(args.model_path.rstrip("/"))
        args.output_dir = f"./eval_results/{model_name_clean}"
        print(f"Auto-generated output directory: {args.output_dir}")


    utils.judge.VLLM_API_BASE = args.judge_api_base
    utils.judge.JUDGE_MODEL = args.judge_model


    TASK_DATA_MAP = {
        key: TASK_DATA_REL_PATHS[key]
        for key in [
            "visual_grounding",
            "attribute_recognition",
            "scene_reasoning",
            "style_aware_caption",
            "aesthetic_assessment",
            "forgery_detection",
            "fast_concept_mapping",
            "world_aware_planning",
            "analogical_inference",
        ]
    }


    TASK_CONFIG = {
        "visual_grounding": eval_grounding_qwen3vl,
        "attribute_recognition": evaluators.eval_attr_rec_gen,
        "scene_reasoning": evaluators.eval_vqa_gen,
        "style_aware_caption": evaluators.eval_caption_styled,
        "aesthetic_assessment": evaluators.eval_aesthetic_assessment,
        "forgery_detection": evaluators.eval_authenticity_detection,
        "fast_concept_mapping": evaluators.eval_fcb_classification,
        "world_aware_planning": evaluators.eval_planning,
        "analogical_inference": evaluators.eval_visualcloze_u,
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

    inferencer = Qwen3VLInferencer(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    def run_task(task_name: str, data_path: str, image_dir: str):
        if task_name not in TASK_CONFIG:
            print(f"⚠️  Skipping {task_name}: unsupported")
            return
        eval_func = TASK_CONFIG[task_name]
        if not os.path.exists(data_path):
            print(f"⚠️  Skipping {task_name}: data file not found at {data_path}")
            return
        output_path = os.path.join(output_dir, f"{task_name}_results.json")
        eval_func(inferencer, data_path, image_dir, output_path, args.k_shot)

    if args.task == "all":

        parser.error("--task all is not supported. Use run_eval.sh to run all tasks with correct image directories.")
    else:
        data_path = args.data_path or os.path.join(args.benchmark_dir, TASK_DATA_MAP.get(args.task, ""))
        run_task(args.task, data_path, args.image_dir)

    print(f"\n✅ Evaluation completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
