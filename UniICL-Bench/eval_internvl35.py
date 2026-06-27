"""
InternVL3.5-8B UniICL-Bench Evaluation (Understanding-only)
--------------------------------------------------------
Uses lmdeploy pipeline for inference. Generation-style tasks are skipped.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
from public_path_config import DEFAULT_JUDGE_MODEL, TASK_DATA_REL_PATHS, normalize_task_name

# Add project roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import evaluators
import utils.judge  # noqa: E402


class InternVL35Inferencer:
    """Minimal wrapper exposing interleave_inference for understanding tasks."""

    def __init__(
        self,
        model: str,
        tp: int = 1,
        session_len: int = 32768,
        temperature: float = 0.1,
        max_new_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 50,
    ):
        backend_config = PytorchEngineConfig(
            session_len=session_len, 
            tp=tp,
            block_size=32,
            max_batch_size=1,
            cache_max_entry_count=0.8,
        )
        self.pipe = pipeline(model, backend_config=backend_config)
        self.gen_config = GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
        )

    def interleave_inference(self, input_lists: List[Any], understanding_output: bool, **kwargs):
        if not understanding_output:
            raise ValueError("InternVL3.5 wrapper only supports understanding_output=True")

        texts: List[str] = []
        images = []
        img_count = 0

        for term in input_lists:
            if isinstance(term, str):
                texts.append(term)
            else:
                img_count += 1
                placeholder = f"Image-{img_count}: {IMAGE_TOKEN}"
                texts.append(placeholder)
                img_path = None
                if hasattr(term, "filename") and term.filename:
                    img_path = term.filename
                if not img_path:
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        term.save(tmp.name)
                        img_path = tmp.name
                images.append(load_image(img_path))

        prompt = "\n".join(texts) if texts else "Describe the images."
        resp = self.pipe((prompt, images), gen_config=self.gen_config)
        return [resp.text]


def main():
    parser = argparse.ArgumentParser(description="InternVL3.5-8B Understanding UniICL-Bench Evaluation")
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3_5-8B", help="Model name or path")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel for lmdeploy (tp=2 for 38B, tp=8 for 241B-A28B)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum new tokens")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--judge-api-base", type=str, default="http://localhost:8000/v1", help="vLLM API base URL for judge model")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL, help="Judge model name for evaluation")
    parser.add_argument("--task", type=str, default="all",
                        help="Paper-aligned task name in snake_case, or all")
    parser.add_argument("--data-path", type=str, help="Task data path (auto when not set)")
    parser.add_argument("--image-dir", type=str, help="Image root (auto when not set)")
    parser.add_argument("--benchmark-dir", type=str, default=".", help="UniICL-Bench root directory")
    parser.add_argument("--output-dir", type=str, default="./eval_results_internvl", help="Output directory")
    parser.add_argument("--k-shot", type=int, default=0, help="ICL k-shot setting")
    args = parser.parse_args()
    try:
        args.task = normalize_task_name(args.task, allow_all=True)
    except ValueError as e:
        parser.error(str(e))


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
        "visual_grounding": evaluators.eval_grounding,
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

    inferencer = InternVL35Inferencer(
        model=args.model,
        tp=args.tp,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
    )


    class GroundingInferencer:
        """Wraps InternVL35Inferencer for grounding: appends coordinate hint to the last text item."""
        COORD_HINT = "Output only normalized [0,1] coordinates in the format [x1, y1, x2, y2]. Do not output anything else."

        def __init__(self, base):
            self._base = base

        def interleave_inference(self, input_lists, **kwargs):
            # Find the last text item and append the hint
            patched = list(input_lists)
            for i in range(len(patched) - 1, -1, -1):
                if isinstance(patched[i], str):
                    patched[i] = patched[i].rstrip() + " " + self.COORD_HINT
                    break
            return self._base.interleave_inference(patched, **kwargs)

    grounding_inferencer = GroundingInferencer(inferencer)

    def run_task(task_name: str, data_path: str, image_dir: str):
        if task_name not in TASK_CONFIG:
            print(f"⚠️  Skipping {task_name}: unsupported")
            return
        eval_func = TASK_CONFIG[task_name]
        if not os.path.exists(data_path):
            print(f"⚠️  Skipping {task_name}: data file not found at {data_path}")
            return
        output_path = os.path.join(output_dir, f"{task_name}_results.json")

        _inferencer = grounding_inferencer if task_name == "visual_grounding" else inferencer
        eval_func(_inferencer, data_path, image_dir, output_path, args.k_shot)

    if args.task == "all":

        parser.error("--task all is not supported. Use run_eval.sh to run all tasks with correct image directories.")
    else:
        data_path = args.data_path or os.path.join(args.benchmark_dir, TASK_DATA_MAP.get(args.task, ""))
        run_task(args.task, data_path, args.image_dir)

    print(f"\n✅ Evaluation completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
