#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "UniICL-Bench"))
sys.path.insert(0, str(PROJECT_ROOT / "UniICL"))

from eval_uniicl import SafeInterleaveInferencer, load_uniicl_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal UniICL inference entrypoint.")
    parser.add_argument("--model-path", required=True, help="Path to the UniICL finetuned checkpoint.")
    parser.add_argument("--base-model-path", default=None, help="Optional base model path for mixed-weight loading.")
    parser.add_argument("--prompt", default=None, help="Input text prompt.")
    parser.add_argument("--image", default=None, help="Optional input image.")
    parser.add_argument("--output-image", default=None, help="Where to save a generated image.")
    parser.add_argument("--understanding-output", action="store_true", help="Return text output instead of generating an image.")
    parser.add_argument("--think", action="store_true", help="Enable think-then-answer / think-then-generate mode.")
    parser.add_argument("--do-sample", action="store_true", help="Sample text outputs instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Text decoding temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum text tokens for understanding or thinking.")
    parser.add_argument("--use-mixed-weights", action="store_true", help="Load delta checkpoint on top of a base model.")
    parser.add_argument("--no-capm", action="store_true", help="Disable CAPM at inference time.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.prompt is None and args.image is None:
        raise ValueError("Please provide at least one of --prompt or --image.")

    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_uniicl_model(
        args.model_path,
        base_model_path=args.base_model_path,
        use_mixed_weights=args.use_mixed_weights,
        enable_capm=not args.no_capm,
    )
    inferencer = SafeInterleaveInferencer(
        model,
        vae_model,
        tokenizer,
        vae_transform,
        vit_transform,
        new_token_ids,
    )

    input_image = None
    if args.image:
        input_image = Image.open(args.image).convert("RGB")

    result = inferencer(
        image=input_image,
        text=args.prompt,
        think=args.think,
        understanding_output=args.understanding_output,
        max_think_token_n=args.max_new_tokens,
        do_sample=args.do_sample,
        text_temperature=args.temperature,
    )

    if result.get("text"):
        print(result["text"])

    if result.get("image") is not None:
        output_path = args.output_image or "uniicl_generation.png"
        result["image"].save(output_path)
        print(f"Saved generated image to {output_path}")


if __name__ == "__main__":
    main()
