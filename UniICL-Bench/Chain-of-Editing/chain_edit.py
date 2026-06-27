"""Public release module documentation."""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional

import torch
from PIL import Image
from diffusers import Flux2Pipeline

from public_path_config import DEFAULT_FLUX_MODEL


def parse_args():
    parser = argparse.ArgumentParser(description="Chain-of-Editing Pipeline")
    parser.add_argument("--data", type=str, default="chain_of_editing_benchmark.json", 
                        help="Path to chain_of_editing_benchmark.json")
    parser.add_argument("--output", type=str, default="Chain-of-Editing",
                        help="Output directory for generated images")
    parser.add_argument("--flux_model_path", type=str, default=DEFAULT_FLUX_MODEL,
                        help="Path or HuggingFace repo for FLUX.2-dev model")
    parser.add_argument("--start_id", type=int, default=None,
                        help="Start sample ID (inclusive)")
    parser.add_argument("--end_id", type=int, default=None,
                        help="End sample ID (inclusive)")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip original image generation (use existing images)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip samples that already have all images")
    parser.add_argument("--gen_steps", type=int, default=28,
                        help="Number of inference steps for generation")
    parser.add_argument("--edit_steps", type=int, default=28,
                        help="Number of inference steps for editing")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="Guidance scale for generation and editing")
    parser.add_argument("--image_size", type=int, default=1024,
                        help="Generated image size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    return parser.parse_args()


class ChainOfEditingPipeline:
    """Chain-of-Editing Pipeline using FLUX.2-dev"""
    
    def __init__(
        self,
        flux_model_path: str,
        device: str = "cuda",
        torch_dtype = torch.bfloat16
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.flux_model_path = flux_model_path
        
        self.pipe = None
    
    def load_model(self):
        """Public release documentation."""
        if self.pipe is not None:
            return
        
        print(f"Loading FLUX.2-dev from {self.flux_model_path}...")
        
        self.pipe = Flux2Pipeline.from_pretrained(
            self.flux_model_path,
            torch_dtype=self.torch_dtype,
        )
        self.pipe = self.pipe.to(self.device)
        print("FLUX.2-dev loaded successfully!")
    
    def unload_model(self):
        """Public release documentation."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            print("FLUX.2-dev unloaded.")
    
    def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Public release documentation."""
        self.load_model()
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        return image
    
    def edit_image(
        self,
        image: Image.Image,
        instruction: str,
        num_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Public release documentation."""
        self.load_model()
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        edited_image = self.pipe(
            prompt=instruction,
            image=image,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        return edited_image
    
    def process_sample(
        self,
        sample: Dict,
        output_dir: Path,
        gen_steps: int = 28,
        edit_steps: int = 28,
        guidance_scale: float = 3.5,
        image_size: int = 1024,
        seed: int = 42,
        skip_generation: bool = False,
        skip_existing: bool = False
    ) -> Dict:
        """Public release documentation."""
        sample_id = sample["id"]
        original_prompt = sample["original_t2i_prompt"]
        edit_history = sample.get("edit_history", [])
        final_instruction = sample["final_instruction"]
        

        all_edits = edit_history + [final_instruction]
        total_steps = len(all_edits) + 1  # +1 for original image
        
        print(f"\n{'='*60}")
        print(f"Processing Sample {sample_id} ({sample['type']})")
        print(f"Edit chain length: {len(all_edits)}")
        print(f"{'='*60}")
        
        result = {
            "id": sample_id,
            "type": sample["type"],
            "images": [],
            "timings": {}
        }
        

        original_path = output_dir / f"sample_{sample_id}_0.jpg"
        
        if skip_generation and original_path.exists():
            print(f"[0/{total_steps-1}] Loading existing original image...")
            current_image = Image.open(original_path).convert("RGB")
        elif original_path.exists() and skip_existing:
            print(f"[0/{total_steps-1}] Skipping existing original image...")
            current_image = Image.open(original_path).convert("RGB")
        else:
            print(f"[0/{total_steps-1}] Generating original image...")
            print(f"  Prompt: {original_prompt[:100]}...")
            
            start_time = time.time()
            current_image = self.generate_image(
                prompt=original_prompt,
                width=image_size,
                height=image_size,
                num_steps=gen_steps,
                seed=seed
            )
            gen_time = time.time() - start_time
            result["timings"]["generation"] = gen_time
            print(f"  Generated in {gen_time:.2f}s")
            
            current_image.save(original_path, quality=95)
        
        result["images"].append(str(original_path))
        

        edit_times = []
        for i, edit_instruction in enumerate(all_edits, 1):
            output_path = output_dir / f"sample_{sample_id}_{i}.jpg"
            
            if output_path.exists() and skip_existing:
                print(f"[{i}/{total_steps-1}] Skipping existing: {edit_instruction[:50]}...")
                current_image = Image.open(output_path).convert("RGB")
            else:
                print(f"[{i}/{total_steps-1}] Editing: {edit_instruction}")
                
                start_time = time.time()
                current_image = self.edit_image(
                    image=current_image,
                    instruction=edit_instruction,
                    num_steps=edit_steps,
                    guidance_scale=guidance_scale,
                    seed=seed + i
                )
                edit_time = time.time() - start_time
                edit_times.append(edit_time)
                print(f"  Edited in {edit_time:.2f}s")
                
                current_image.save(output_path, quality=95)
            
            result["images"].append(str(output_path))
        
        if edit_times:
            result["timings"]["edits"] = edit_times
            result["timings"]["total_edit_time"] = sum(edit_times)
        
        print(f"Sample {sample_id} completed!")
        return result


def load_data(data_path: str) -> List[Dict]:
    """Public release documentation."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main():
    args = parse_args()
    
    print("="*60)
    print("Chain-of-Editing Pipeline (FLUX.2-dev)")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"FLUX model path: {args.flux_model_path}")
    print(f"Generation steps: {args.gen_steps}")
    print(f"Edit steps: {args.edit_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Image size: {args.image_size}")
    print(f"Seed: {args.seed}")
    print("="*60)
    

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    

    data = load_data(args.data)
    print(f"Loaded {len(data)} samples")
    

    if args.start_id is not None or args.end_id is not None:
        start_id = args.start_id or 1
        end_id = args.end_id or max(s["id"] for s in data)
        data = [s for s in data if start_id <= s["id"] <= end_id]
        print(f"Processing samples {start_id} to {end_id} ({len(data)} samples)")
    

    pipeline = ChainOfEditingPipeline(
        flux_model_path=args.flux_model_path,
        device=args.device,
        torch_dtype=torch.bfloat16
    )
    
    results = {s["id"]: {"id": s["id"], "type": s["type"], "images": [], "timings": {}} for s in data}
    total_start = time.time()
    

    if not args.skip_generation:
        print("\n" + "="*60)
        print("Phase 1: Generating Original Images (FLUX.2-dev)")
        print("="*60)
        
        pipeline.load_model()
        gen_times = []
        
        for i, sample in enumerate(data):
            sample_id = sample["id"]
            original_path = output_dir / f"sample_{sample_id}_0.jpg"
            
            if original_path.exists() and args.skip_existing:
                print(f"[{i+1}/{len(data)}] Sample {sample_id}: Skipping (exists)")
                results[sample_id]["images"].append(str(original_path))
                continue
            
            print(f"[{i+1}/{len(data)}] Sample {sample_id}: Generating...")
            
            try:
                start_time = time.time()
                image = pipeline.generate_image(
                    prompt=sample["original_t2i_prompt"],
                    width=args.image_size,
                    height=args.image_size,
                    num_steps=args.gen_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed
                )
                gen_time = time.time() - start_time
                gen_times.append(gen_time)
                
                image.save(original_path, quality=95)
                results[sample_id]["images"].append(str(original_path))
                results[sample_id]["timings"]["generation"] = gen_time
                print(f"  Done in {gen_time:.2f}s")
                
            except Exception as e:
                print(f"  Error: {e}")
                results[sample_id]["error"] = str(e)
        
        if gen_times:
            print(f"\nPhase 1 complete: {len(gen_times)} images, avg {sum(gen_times)/len(gen_times):.2f}s/image")
    else:
        print("\n[Skipping Phase 1: Using existing original images]")
        for sample in data:
            sample_id = sample["id"]
            original_path = output_dir / f"sample_{sample_id}_0.jpg"
            if original_path.exists():
                results[sample_id]["images"].append(str(original_path))
    

    print("\n" + "="*60)
    print("Phase 2: Editing Chain (FLUX.2-dev)")
    print("="*60)
    
    pipeline.load_model()
    
    for i, sample in enumerate(data):
        sample_id = sample["id"]
        edit_history = sample.get("edit_history", [])
        final_instruction = sample["final_instruction"]
        all_edits = edit_history + [final_instruction]
        

        if "error" in results[sample_id]:
            print(f"[{i+1}/{len(data)}] Sample {sample_id}: Skipping (generation failed)")
            continue
        
        original_path = output_dir / f"sample_{sample_id}_0.jpg"
        if not original_path.exists():
            print(f"[{i+1}/{len(data)}] Sample {sample_id}: Skipping (no original image)")
            continue
        
        print(f"\n[{i+1}/{len(data)}] Sample {sample_id}: {len(all_edits)} edits")
        

        current_image = Image.open(original_path).convert("RGB")
        edit_times = []
        
        for j, edit_instruction in enumerate(all_edits, 1):
            output_path = output_dir / f"sample_{sample_id}_{j}.jpg"
            
            if output_path.exists() and args.skip_existing:
                print(f"  [{j}/{len(all_edits)}] Skipping: {edit_instruction[:40]}...")
                current_image = Image.open(output_path).convert("RGB")
                results[sample_id]["images"].append(str(output_path))
                continue
            
            print(f"  [{j}/{len(all_edits)}] {edit_instruction[:50]}...")
            
            try:
                start_time = time.time()
                current_image = pipeline.edit_image(
                    image=current_image,
                    instruction=edit_instruction,
                    num_steps=args.edit_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed + j
                )
                edit_time = time.time() - start_time
                edit_times.append(edit_time)
                
                current_image.save(output_path, quality=95)
                results[sample_id]["images"].append(str(output_path))
                print(f"    Done in {edit_time:.2f}s")
                
            except Exception as e:
                print(f"    Error: {e}")
                results[sample_id]["error"] = str(e)
                break
        
        if edit_times:
            results[sample_id]["timings"]["edits"] = edit_times
            results[sample_id]["timings"]["total_edit_time"] = sum(edit_times)
    
    total_time = time.time() - total_start
    

    results_list = list(results.values())
    stats = {
        "total_samples": len(data),
        "successful": len([r for r in results_list if "error" not in r]),
        "failed": len([r for r in results_list if "error" in r]),
        "total_time": total_time,
        "average_time_per_sample": total_time / len(data) if data else 0,
        "results": results_list
    }
    
    stats_path = output_dir / "processing_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per sample: {stats['average_time_per_sample']:.1f}s")
    print(f"Results saved to: {stats_path}")


if __name__ == "__main__":
    main()
"""
# Use the FLUX.2-dev model
    python chain_edit.py \
        --data chain_of_editing_benchmark.json \
        --output images \
        --flux_model_path /path/to/flux-model \
        --start_id 1 \
        --end_id 50 \
        --gen_steps 28 \
        --edit_steps 28 \
        --guidance_scale 3.5

# Skip images that already exist
python chain_edit.py \
    --data chain_of_editing_benchmark.json \
    --output Chain-of-Editing \
    --flux_model_path /path/to/flux-model \
    --skip_existing
"""
