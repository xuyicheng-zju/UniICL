"""Public release module documentation."""

import json
import argparse
import requests
import base64
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Chain-of-Editing Quality")
    parser.add_argument("--data", type=str, default="chain_of_editing_benchmark.json",
                        help="Path to chain_of_editing_benchmark.json")
    parser.add_argument("--image_dir", type=str, default="images",
                        help="Directory containing generated images")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output JSON file for evaluation results")
    parser.add_argument("--vllm_url", type=str,
                        default="http://localhost:8011/v1/chat/completions",
                        help="vLLM API endpoint URL")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2-VL-72B-Instruct",
                        help="Model name for vLLM")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for MLLM")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of top samples to select")
    return parser.parse_args()


def encode_image_to_base64(image_path: str) -> str:
    """Public release documentation."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vllm_judge(
    prompt: str,
    image_paths: List[str],
    vllm_url: str,
    model: str,
    temperature: float = 0.0
) -> str:
    """Public release documentation."""


    content = []


    for img_path in image_paths:
        img_base64 = encode_image_to_base64(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
        })


    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": 1024
    }

    try:
        response = requests.post(vllm_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling vLLM: {e}")
        return ""


def evaluate_single_edit(
    before_image: str,
    after_image: str,
    edit_instruction: str,
    vllm_url: str,
    model: str,
    temperature: float = 0.0
) -> Dict:
    """Public release documentation."""

    prompt = f"""You are an expert image editing evaluator. Compare the BEFORE and AFTER images to assess the quality of this edit.

**Edit Instruction**: {edit_instruction}

Evaluate the edit on THREE dimensions:

**1. Edit Correctness (0-40 points)**
Did the edit correctly execute ALL aspects of the instruction?
- 35-40: Perfect execution, all requirements met precisely
- 25-34: Mostly correct, minor deviations or imprecisions
- 15-24: Partially correct, some requirements missing or wrong
- 5-14: Major errors or significant omissions
- 0-4: Failed to execute or completely wrong

**2. Preservation Quality (0-40 points)**
Were ALL regions NOT mentioned in the instruction perfectly preserved?
- 35-40: Perfect preservation, no unintended changes
- 25-34: Minor unintended changes in non-target areas
- 15-24: Noticeable unintended changes
- 5-14: Significant unintended modifications
- 0-4: Widespread unintended changes

**3. Instruction Compliance (0-20 points)**
Did the edit follow all detailed requirements (lighting, shadows, reflections, textures, consistency)?
- 18-20: Perfect compliance with all details
- 14-17: Mostly compliant, minor detail issues
- 10-13: Partial compliance, some details incorrect
- 5-9: Poor compliance, many details wrong
- 0-4: Failed to comply with detail requirements

**Output Format** (be strict and precise):
Rationale: [2-3 sentences explaining your assessment]
Edit_Correctness: [score 0-40]
Preservation_Quality: [score 0-40]
Instruction_Compliance: [score 0-20]
Total_Score: [sum of three scores, 0-100]"""

    response = call_vllm_judge(prompt, [before_image, after_image], vllm_url, model, temperature)


    result = {
        "rationale": "",
        "edit_correctness": 0,
        "preservation_quality": 0,
        "instruction_compliance": 0,
        "total_score": 0,
        "raw_response": response
    }

    try:

        rationale_match = re.search(r'Rationale[:\s]+(.+?)(?=Edit_Correctness|$)', response, re.IGNORECASE | re.DOTALL)
        if rationale_match:
            result["rationale"] = rationale_match.group(1).strip()


        edit_match = re.search(r'Edit[_\s]*Correctness[:\s]+(\d+)', response, re.IGNORECASE)
        pres_match = re.search(r'Preservation[_\s]*Quality[:\s]+(\d+)', response, re.IGNORECASE)
        comp_match = re.search(r'Instruction[_\s]*Compliance[:\s]+(\d+)', response, re.IGNORECASE)
        total_match = re.search(r'Total[_\s]*Score[:\s]+(\d+)', response, re.IGNORECASE)

        if edit_match:
            result["edit_correctness"] = min(40, max(0, int(edit_match.group(1))))
        if pres_match:
            result["preservation_quality"] = min(40, max(0, int(pres_match.group(1))))
        if comp_match:
            result["instruction_compliance"] = min(20, max(0, int(comp_match.group(1))))
        if total_match:
            result["total_score"] = min(100, max(0, int(total_match.group(1))))


        if result["total_score"] == 0:
            result["total_score"] = (result["edit_correctness"] +
                                    result["preservation_quality"] +
                                    result["instruction_compliance"])

    except Exception as e:
        print(f"Error parsing scores: {e}")

    return result


def evaluate_sample(
    sample: Dict,
    image_dir: Path,
    vllm_url: str,
    model: str,
    temperature: float = 0.0
) -> Dict:
    """Public release documentation."""

    sample_id = sample["id"]
    sample_type = sample["type"]
    edit_history = sample.get("edit_history", [])
    final_instruction = sample["final_instruction"]
    all_edits = edit_history + [final_instruction]


    image_paths = []
    for i in range(len(all_edits) + 1):  # +1 for original
        img_path = image_dir / f"sample_{sample_id}_{i}.jpg"
        if not img_path.exists():
            return {
                "id": sample_id,
                "type": sample_type,
                "status": "missing_images",
                "missing_path": str(img_path)
            }
        image_paths.append(img_path)


    step_evaluations = []
    total_score = 0

    for i, edit_instruction in enumerate(all_edits):
        before_img = str(image_paths[i])
        after_img = str(image_paths[i + 1])

        print(f"  Evaluating step {i+1}/{len(all_edits)}: {edit_instruction[:50]}...")

        eval_result = evaluate_single_edit(
            before_image=before_img,
            after_image=after_img,
            edit_instruction=edit_instruction,
            vllm_url=vllm_url,
            model=model,
            temperature=temperature
        )

        step_evaluations.append({
            "step": i + 1,
            "instruction": edit_instruction,
            "before_image": before_img,
            "after_image": after_img,
            **eval_result
        })

        total_score += eval_result["total_score"]


    avg_score = total_score / len(all_edits) if all_edits else 0


    avg_edit_correctness = sum(s["edit_correctness"] for s in step_evaluations) / len(step_evaluations)
    avg_preservation = sum(s["preservation_quality"] for s in step_evaluations) / len(step_evaluations)
    avg_compliance = sum(s["instruction_compliance"] for s in step_evaluations) / len(step_evaluations)

    return {
        "id": sample_id,
        "type": sample_type,
        "status": "evaluated",
        "num_edits": len(all_edits),
        "step_evaluations": step_evaluations,
        "total_score": total_score,
        "average_score": avg_score,
        "avg_edit_correctness": avg_edit_correctness,
        "avg_preservation_quality": avg_preservation,
        "avg_instruction_compliance": avg_compliance
    }


def select_top_samples(
    evaluations: List[Dict],
    top_k: int = 50,
    prioritize_long: bool = True
) -> List[Dict]:
    """Public release documentation."""


    valid_samples = [e for e in evaluations if e.get("status") == "evaluated"]

    if not valid_samples:
        return []


    medium_samples = [s for s in valid_samples if s["type"] == "Medium"]
    long_samples = [s for s in valid_samples if s["type"] == "Long"]


    medium_samples.sort(key=lambda x: x["average_score"], reverse=True)
    long_samples.sort(key=lambda x: x["average_score"], reverse=True)

    if not prioritize_long:

        valid_samples.sort(key=lambda x: x["average_score"], reverse=True)
        return valid_samples[:top_k]






    selected = []


    high_quality_long = [s for s in long_samples if s["average_score"] >= 70]
    selected.extend(high_quality_long)
    print(f"Selected {len(high_quality_long)} high-quality Long samples (>=70)")

    if len(selected) >= top_k:
        return selected[:top_k]


    remaining = top_k - len(selected)
    high_quality_medium = [s for s in medium_samples if s["average_score"] >= 70]
    selected.extend(high_quality_medium[:remaining])
    print(f"Added {min(len(high_quality_medium), remaining)} high-quality Medium samples (>=70)")

    if len(selected) >= top_k:
        return selected[:top_k]


    remaining = top_k - len(selected)
    remaining_long = [s for s in long_samples if s not in selected]
    selected.extend(remaining_long[:remaining])
    print(f"Added {min(len(remaining_long), remaining)} additional Long samples")

    if len(selected) >= top_k:
        return selected[:top_k]


    remaining = top_k - len(selected)
    remaining_medium = [s for s in medium_samples if s not in selected]
    selected.extend(remaining_medium[:remaining])
    print(f"Added {min(len(remaining_medium), remaining)} additional Medium samples")

    return selected[:top_k]


def main():
    args = parse_args()

    print("="*60)
    print("Chain-of-Editing Quality Evaluation")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Image directory: {args.image_dir}")
    print(f"MLLM: {args.model}")
    print(f"Top-K: {args.top_k}")
    print("="*60)


    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")


    medium_count = len([s for s in data if s["type"] == "Medium"])
    long_count = len([s for s in data if s["type"] == "Long"])
    print(f"  Medium: {medium_count}")
    print(f"  Long: {long_count}")

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return


    print("\nEvaluating all samples...")
    evaluations = []

    for sample in tqdm(data, desc="Evaluating samples"):
        sample_id = sample["id"]
        print(f"\nSample {sample_id} ({sample['type']}):")

        eval_result = evaluate_sample(
            sample=sample,
            image_dir=image_dir,
            vllm_url=args.vllm_url,
            model=args.model,
            temperature=args.temperature
        )

        evaluations.append(eval_result)

        if eval_result.get("status") == "evaluated":
            print(f"  Average Score: {eval_result['average_score']:.2f}/100")
            print(f"    Edit Correctness: {eval_result['avg_edit_correctness']:.2f}/40")
            print(f"    Preservation: {eval_result['avg_preservation_quality']:.2f}/40")
            print(f"    Compliance: {eval_result['avg_instruction_compliance']:.2f}/20")
        else:
            print(f"  Status: {eval_result.get('status')}")


    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(evaluations, f, indent=2, ensure_ascii=False)

    print(f"\nFull evaluation results saved to: {args.output}")


    print(f"\nSelecting top-{args.top_k} samples (prioritizing Long type)...")
    top_samples = select_top_samples(evaluations, args.top_k, prioritize_long=True)


    selected_medium = len([s for s in top_samples if s["type"] == "Medium"])
    selected_long = len([s for s in top_samples if s["type"] == "Long"])

    print(f"\nSelected {len(top_samples)} samples:")
    print(f"  Medium: {selected_medium}")
    print(f"  Long: {selected_long} ({selected_long/len(top_samples)*100:.1f}%)")

    if top_samples:
        avg_score = sum(s["average_score"] for s in top_samples) / len(top_samples)
        min_score = min(s["average_score"] for s in top_samples)
        max_score = max(s["average_score"] for s in top_samples)
        print(f"  Score range: {min_score:.2f} - {max_score:.2f}")
        print(f"  Average score: {avg_score:.2f}")


    selected_output = args.output.replace(".json", "_top50.json")
    with open(selected_output, 'w', encoding='utf-8') as f:
        json.dump(top_samples, f, indent=2, ensure_ascii=False)

    print(f"\nTop-{args.top_k} samples saved to: {selected_output}")


    selected_ids = {s["id"] for s in top_samples}
    benchmark_data = [s for s in data if s["id"] in selected_ids]


    benchmark_data.sort(key=lambda x: x["id"])

    benchmark_output = args.output.replace(".json", "_benchmark.json")
    with open(benchmark_output, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print(f"UniICL-Bench data (selected samples only) saved to: {benchmark_output}")


    print(f"\nSelected sample IDs:")
    for s in sorted(top_samples, key=lambda x: x["id"]):
        print(f"  {s['id']:3d} ({s['type']:6s}): {s['average_score']:.2f}")


if __name__ == "__main__":
    main()
