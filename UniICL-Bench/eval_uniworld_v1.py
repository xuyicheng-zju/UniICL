"""
UniWorld-V1 UniICL-Bench Evaluation (ICL UniICL-Bench)
-------------------------------------------------
This script reuses the task evaluation logic from ``eval_bagel.py`` while
swapping in the UniWorld-V1 inference stack. The only requirement for the
inferencer is to expose ``interleave_inference`` that accepts the same
arguments as Bagel's version.
"""
import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNI_ROOT = PROJECT_ROOT / "UniWorld" / "UniWorld-V1"

# Make sure benchmark can import UniWorld and base eval utils
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(UNI_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse all evaluation utilities from the Bagel script
from utils import evaluators
import utils.judge  # noqa: E402
from public_path_config import DEFAULT_HPSV3_CHECKPOINT, DEFAULT_JUDGE_MODEL, DEFAULT_QALIGN_MODEL, TASK_DATA_REL_PATHS, normalize_task_name

# UniWorld imports
from qwen_vl_utils import process_vision_info  # noqa: E402
from univa.serve.infer_icl import (  # noqa: E402
    load_main_model_and_processor,
    load_pipe,
    load_siglip_and_processor,
    preprocess_siglip_pixel_values,
    update_size,
)
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt  # noqa: E402


class UniWorldICLInferencer:
    """Thin compatibility wrapper so Bagel evaluators can call UniWorld."""

    def __init__(
        self,
        model_path: str,
        flux_path: str,
        siglip_path: str = "",
        device: torch.device | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.task_head, self.processor = load_main_model_and_processor(
            model_path, self.device
        )

        # Set pad_token_id for open-end generation
        if getattr(self.processor, "tokenizer", None) and self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

        self.pipe, self.tokenizers, self.text_encoders = load_pipe(
            self.model.denoise_tower.denoiser, flux_path, self.device
        )
        self.siglip_processor, self.siglip_model = load_siglip_and_processor(siglip_path, self.device)

        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = 42

    @staticmethod
    def _image_content(path: str) -> Dict[str, Any]:
        return {
            "type": "image",
            "image": path,
            "min_pixels": 448 * 448,
            "max_pixels": 448 * 448,
        }

    @staticmethod
    def _save_temp_image(image: Image.Image, tmpdir: str, idx: int) -> str:
        path = os.path.join(tmpdir, f"img_{idx:04d}.png")
        image.save(path)
        return path

    @staticmethod
    def _split_user_assistant(text: str) -> Tuple[str, str, bool]:
        """Parse a chunk like `User: ... Assistant: ...`."""
        has_assistant = bool(re.search(r"assistant\s*:", text, re.IGNORECASE))
        cleaned = text
        user_part = ""
        assistant_part = ""

        match = re.search(r"User:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if match:
            cleaned = match.group(1)

        parts = re.split(r"Assistant\s*:", cleaned, flags=re.IGNORECASE)
        if parts:
            user_part = parts[0].strip()
            if len(parts) > 1:
                assistant_part = "Assistant:".join(parts[1:]).strip()
        else:
            user_part = cleaned.strip()
        return user_part, assistant_part, has_assistant

    def _build_conversation(
        self, input_lists: List[Any], tmpdir: str, understanding_output: bool
    ) -> List[Dict[str, Any]]:
        """Convert Bagel-style interleave inputs to UniWorld chat format."""
        conversation: List[Dict[str, Any]] = []
        pending_user_images: List[str] = []
        expect_assistant_image = False
        img_idx = 0

        for term in input_lists:
            if isinstance(term, Image.Image):
                img_path = self._save_temp_image(term, tmpdir, img_idx)
                img_idx += 1
                if expect_assistant_image:
                    conversation.append({"role": "assistant", "content": [self._image_content(img_path)]})
                    expect_assistant_image = False
                else:
                    pending_user_images.append(img_path)
                continue

            if isinstance(term, str):
                user_text, assistant_text, has_assistant = self._split_user_assistant(term)

                if user_text or pending_user_images:
                    # Preserve the original arrival order: images were seen before this text.
                    content = []
                    content.extend(self._image_content(p) for p in pending_user_images)
                    if user_text:
                        content.append({"type": "text", "text": user_text})
                    conversation.append({"role": "user", "content": content})
                    pending_user_images = []

                if assistant_text:
                    conversation.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
                    expect_assistant_image = False
                elif has_assistant and not understanding_output:
                    # Next image (if any) is treated as assistant response demo
                    expect_assistant_image = True

        if pending_user_images:
            default_prompt = (
                "Generate the output image following previous examples."
                if not understanding_output
                else "Please respond based on these images."
            )
            content = []
            content.extend(self._image_content(p) for p in pending_user_images)
            content.append({"type": "text", "text": default_prompt})
            conversation.append({"role": "user", "content": content})

        if not conversation or conversation[-1].get("role") != "user":
            conversation.append({"role": "user", "content": [{"type": "text", "text": "Please provide the final response."}]})

        return conversation

    def _prepare_inputs(self, conversation: List[Dict[str, Any]]):
        chat_text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        # Drop system token to stay close to Bagel's behavior
        chat_text = "<|im_end|>\n".join(chat_text.split("<|im_end|>\n")[1:])
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.device)

    @staticmethod
    def _extract_last_user_text(conversation: List[Dict[str, Any]]) -> str:
        for msg in reversed(conversation):
            if msg.get("role") != "user":
                continue
            for content in msg.get("content", []):
                if content.get("type") == "text":
                    return content.get("text", "")
        return ""

    @staticmethod
    def _collect_image_paths(conversation: List[Dict[str, Any]]) -> List[str]:
        paths = []
        for msg in conversation:
            for content in msg.get("content", []):
                if content.get("type") == "image" and content.get("image"):
                    paths.append(content["image"])
        return paths

    def _run_understanding(
        self,
        conversation: List[Dict[str, Any]],
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        do_sample: bool = False,
    ) -> str:
        inputs = self._prepare_inputs(conversation)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        reply = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return reply.strip()

    def _run_generation(
        self,
        conversation: List[Dict[str, Any]],
        num_timesteps: int | None = None,
        guidance_scale: float | None = None,
        no_joint_with_t5: bool = False,
    ) -> Image.Image:
        inputs = self._prepare_inputs(conversation)

        all_image_paths = self._collect_image_paths(conversation)
        target_h, target_w = self.height, self.width
        new_h, new_w = update_size(
            all_image_paths[:2] if len(all_image_paths) >= 2 else all_image_paths,
            "any_11ratio",
            anchor_pixels=target_h * target_w,
        )

        siglip_hidden_states = None
        if self.siglip_processor is not None and all_image_paths:
            siglip_hidden_states = preprocess_siglip_pixel_values(
                self.siglip_model, self.siglip_processor, all_image_paths
            )

        with torch.inference_mode():
            lvlm_embeds = self.model(
                inputs.input_ids,
                pixel_values=getattr(inputs, "pixel_values", None),
                attention_mask=inputs.attention_mask,
                image_grid_thw=getattr(inputs, "image_grid_thw", None),
                siglip_hidden_states=siglip_hidden_states,
                output_type="denoise_embeds",
            )

        input_embeds = lvlm_embeds
        query_text = self._extract_last_user_text(conversation)

        t5_prompt_embeds, pooled_prompt_embeds = encode_prompt(
            self.text_encoders,
            self.tokenizers,
            query_text if not no_joint_with_t5 else "",
            256,
            self.device,
            1,
        )

        if not no_joint_with_t5:
            input_embeds = torch.concat([t5_prompt_embeds, input_embeds], dim=1)

        steps = num_timesteps or self.num_inference_steps
        scale = guidance_scale if guidance_scale is not None else self.guidance_scale

        output_image = self.pipe(
            prompt_embeds=input_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=new_h,
            width=new_w,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=torch.Generator(device=self.device).manual_seed(self.seed),
        ).images[0]

        return output_image

    def interleave_inference(self, input_lists: List[Any], understanding_output: bool, **kwargs):
        """Compatibility entrypoint used by Bagel evaluators."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conversation = self._build_conversation(input_lists, tmpdir, understanding_output)

            if understanding_output:
                result = self._run_understanding(
                    conversation,
                    max_new_tokens=kwargs.get("max_think_token_n", 2048),
                    temperature=kwargs.get("text_temperature", 0.1),
                    do_sample=kwargs.get("do_sample", False),
                )
            else:
                result = self._run_generation(
                    conversation,
                    num_timesteps=kwargs.get("num_timesteps"),
                    guidance_scale=kwargs.get("cfg_text_scale"),
                    no_joint_with_t5=kwargs.get("no_joint_with_t5", False),
                )

        return [result]


def main():
    parser = argparse.ArgumentParser(description="UniWorld-V1 ICL UniICL-Bench Evaluation")
    parser.add_argument("--model-path", type=str, required=True, help="Path to UniWorld-V1 checkpoint")
    parser.add_argument("--flux-path", type=str, required=True, help="Path to FLUX.1 model")
    parser.add_argument("--siglip-path", type=str, default="", help="Path to SigLIP model (optional)")
    parser.add_argument(
        "--task",
        type=str,
        help="Paper-aligned task name in snake_case, or all",
        default="all",
    )
    parser.add_argument("--data-path", type=str, help="Task data path (auto when not set)")
    parser.add_argument("--image-dir", type=str, help="Image root (auto when not set)")
    parser.add_argument("--benchmark-dir", type=str, default=".", help="UniICL-Bench root directory")
    parser.add_argument("--output-dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--k-shot", type=int, default=0, help="ICL k-shot setting")
    parser.add_argument("--height", type=int, default=1024, help="Generation height")
    parser.add_argument("--width", type=int, default=1024, help="Generation width")
    parser.add_argument("--num-inference-steps", type=int, default=28, help="Diffusion steps for generation")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale for generation")
    parser.add_argument(
        "--hps-checkpoint",
        type=str,
        default=DEFAULT_HPSV3_CHECKPOINT,
        help="HPSv3 checkpoint for scoring",
    )
    parser.add_argument(
        "--judge-api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API base URL for GPT-Score evaluation",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help="Judge model name for GPT-Score",
    )

    args = parser.parse_args()
    try:
        args.task = normalize_task_name(args.task, allow_all=True)
    except ValueError as e:
        parser.error(str(e))

    # Keep base evaluator configs in sync
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
        args.data_path = os.path.join(args.benchmark_dir, TASK_DATA_MAP[args.task])


    if not args.image_dir:
        parser.error("--image-dir is required. Use run_eval.sh for automatic path resolution.")

    if args.k_shot < 0:
        parser.error(f"--k-shot must be non-negative, got {args.k_shot}")


    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    inferencer = UniWorldICLInferencer(
        model_path=args.model_path,
        flux_path=args.flux_path,
        siglip_path=args.siglip_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )

    hps_model = None
    if args.task in ["instructional_generation", "all"]:
        print(f"\nLoading HPSv3 model from {args.hps_checkpoint}...")
        from utils.scoring import load_hpsv3_model
        hps_model = load_hpsv3_model(args.hps_checkpoint)
        if hps_model:
            print("✅ HPSv3 model loaded successfully")
        else:
            print("⚠️ HPSv3 model not loaded, Instructional Generation scoring will be skipped")

    qalign_model = None
    if args.task in ["visual_refinement", "all"]:
        print("\nLoading Q-Align model for Visual Refinement evaluation...")
        from utils.scoring import load_qalign_model
        try:
            qalign_model = load_qalign_model(DEFAULT_QALIGN_MODEL)
            print("✅ Q-Align model loaded successfully")
        except Exception as e:
            print(f"⚠️ Q-Align model not loaded: {e}. Perfection scoring will be skipped")
            qalign_model = None


    if args.task == "all":

        parser.error("--task all is not supported. Use run_eval.sh to run all tasks with correct image directories.")
    else:
        output_path = os.path.join(output_dir, f"{args.task}_results.json")
        if args.task in TASK_CONFIG:
            eval_func, model_type = TASK_CONFIG[args.task]


            class GroundingInferencer:
                COORD_HINT = "Output only normalized [0,1] coordinates in the format [x1, y1, x2, y2]. Do not output anything else."
                def __init__(self, base): self._base = base
                def interleave_inference(self, input_lists, **kwargs):
                    patched = list(input_lists)
                    for i in range(len(patched) - 1, -1, -1):
                        if isinstance(patched[i], str):
                            patched[i] = patched[i].rstrip() + " " + self.COORD_HINT
                            break
                    return self._base.interleave_inference(patched, **kwargs)

            _inferencer = GroundingInferencer(inferencer) if args.task == "visual_grounding" else inferencer

            if args.task == "visual_refinement":
                eval_func(_inferencer, args.data_path, args.image_dir, output_path, args.k_shot, qalign_model=qalign_model)
            elif model_type == "hps":
                eval_func(_inferencer, args.data_path, args.image_dir, output_path, args.k_shot, hps_model=hps_model)
            else:
                eval_func(_inferencer, args.data_path, args.image_dir, output_path, args.k_shot)
        else:
            raise ValueError(f"Unknown task: {args.task}")

    print(f"\n✅ Evaluation completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
