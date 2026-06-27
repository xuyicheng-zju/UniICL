#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Public release module documentation."""

import json
import os
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_IMAGE_BASE_PATH = str(Path(__file__).resolve().parent)


def load_image_bytes(image_path: str) -> bytes:
    """Public release documentation."""
    if not image_path:
        return None
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        # print(f"Warning: Failed to read image {image_path}: {e}")
        return None

def resolve_path(original_path: str, base_path: str) -> str:
    """Public release documentation."""
    if not original_path:
        return ""
    if not base_path:
        return original_path


    if original_path.startswith('/'):
        filename = os.path.basename(original_path)
        return os.path.join(base_path, filename)
    else:

        return os.path.join(base_path, original_path)

def weighted_choice(shots: List[int], weights: List[int]) -> int:
    """Public release documentation."""
    total = sum(weights)
    r = random.randint(1, total)
    cumsum = 0
    for shot, weight in zip(shots, weights):
        cumsum += weight
        if r <= cumsum:
            return shot
    return shots[-1]


def process_t2i_icl_sample(args) -> Dict:
    """Public release documentation."""
    item, image_base_path, shots, weights, pre_sampled = args

    query_instruction = item.get('instruction') or item.get('text', '')
    raw_query_image_path = item.get('answer', '')
    query_image_path = resolve_path(raw_query_image_path, image_base_path)


    available_demos = item.get('demos', [])

    num_demos = weighted_choice(shots, weights)
    selected_demos = available_demos[:num_demos]


    image_list = []
    instruction_list = []

    # Demo samples
    if selected_demos:
        for demo in selected_demos:
            demo_instruction = demo.get('instruction') or demo.get('text', '')
            raw_demo_image_path = demo.get('answer', '')
            demo_image_path = resolve_path(raw_demo_image_path, image_base_path)

            if demo_image_path:
                image_bytes = load_image_bytes(demo_image_path)
                if image_bytes:
                    image_list.append(image_bytes)

                    instruction_list.append(f"Generate an image: {demo_instruction}")

    # Query sample
    if query_image_path:
        query_bytes = load_image_bytes(query_image_path)
        if query_bytes:
            image_list.append(query_bytes)

            instruction_list.append(f"Generate an image: {query_instruction}")


    if len(image_list) >= 1 and len(instruction_list) >= 1:
        return {
            'image_list': image_list,
            'instruction_list': instruction_list
        }
    return None


def process_fci_icl_sample(args) -> Dict:
    """Public release documentation."""
    item, image_base_path, shots, weights, pre_sampled = args

    query_instruction = item.get('instruction') or item.get('text', '')
    raw_query_image_path = item.get('answer', '')
    query_image_path = resolve_path(raw_query_image_path, image_base_path)


    available_demos = item.get('demos', [])

    num_demos = weighted_choice(shots, weights)
    selected_demos = available_demos[:num_demos]

    if not selected_demos:
        return None


    image_list = []
    instruction_list = []
    for demo in selected_demos:
        demo_instruction = demo.get('instruction') or demo.get('text', '')

        raw_demo_image_path = demo.get('answer') or demo.get('image_name', '')
        demo_image_path = resolve_path(raw_demo_image_path, image_base_path)

        if demo_image_path:
            image_bytes = load_image_bytes(demo_image_path)
            if image_bytes:
                image_list.append(image_bytes)

                instruction_list.append(f"Generate an image: {demo_instruction}")

    # Query sample
    if query_image_path:
        query_bytes = load_image_bytes(query_image_path)
        if query_bytes:
            image_list.append(query_bytes)

            instruction_list.append(f"Generate an image: {query_instruction}")

    if len(image_list) >= 2:
        return {
            'image_list': image_list,
            'instruction_list': instruction_list
        }
    return None


def process_i2i_icl_sample(args) -> Dict:
    """Public release documentation."""
    item, image_base_path, shots, weights, pre_sampled = args

    query_source_name = item.get('image_path') or item.get('image_name', '')
    query_instruction = item.get('instruction') or item.get('text', '')
    query_target_name = item.get('answer', '')

    query_source_path = resolve_path(query_source_name, image_base_path)
    query_target_path = resolve_path(query_target_name, image_base_path)

    if not query_source_path:
        return None


    available_demos = item.get('demos', [])

    num_demos = weighted_choice(shots, weights)
    selected_demos = available_demos[:num_demos]


    image_list = []
    instruction_list = []

    # Demo samples
    if selected_demos:
        for demo in selected_demos:
            demo_source_name = demo.get('image_path') or demo.get('image_name', '')
            demo_instruction = demo.get('instruction') or demo.get('text', '')
            demo_target_name = demo.get('answer', '')

            demo_source_path = resolve_path(demo_source_name, image_base_path)
            demo_target_path = resolve_path(demo_target_name, image_base_path)

            if demo_source_path and demo_target_path:
                source_bytes = load_image_bytes(demo_source_path)
                target_bytes = load_image_bytes(demo_target_path)
                if source_bytes and target_bytes:
                    image_list.append(source_bytes)  # demo src
                    image_list.append(target_bytes)  # demo tgt

                    instruction_list.append(demo_instruction)

    # Query sample
    if query_source_path and query_target_path:
        source_bytes = load_image_bytes(query_source_path)
        target_bytes = load_image_bytes(query_target_path)
        if source_bytes and target_bytes:
            image_list.append(source_bytes)  # query src
            image_list.append(target_bytes)  # query tgt

            instruction_list.append(query_instruction)


    if len(instruction_list) >= 1:
        return {
            'image_list': image_list,
            'instruction_list': instruction_list
        }
    return None




def parse_visualcloze_g_task_name(task_name: str) -> Tuple[List[str], str]:
    """Public release documentation."""
    parts = task_name.split('_')
    if len(parts) == 1:
        return [], parts[0]
    return parts[:-1], parts[-1]


def infer_visualcloze_g_fields(item: Dict) -> Tuple[List[str], str]:
    """Public release documentation."""
    task_name = item.get('task_name', '')
    parsed_input_fields, parsed_output_field = parse_visualcloze_g_task_name(task_name) if task_name else ([], "")

    query = item.get('query', {})
    demos = item.get('demo', item.get('demos', []))
    if not isinstance(demos, list):
        demos = []
    if not isinstance(query, dict):
        query = {}

    source_sample = query if query else (demos[0] if demos else {})
    if not isinstance(source_sample, dict):
        source_sample = {}


    output_field = parsed_output_field if parsed_output_field in source_sample else ""


    if not output_field:
        for k in ["output", "reference", "answer", "target", "edited", "edit"]:
            if k in source_sample:
                output_field = k
                break


    if not output_field:
        candidates = [
            k for k, v in source_sample.items()
            if (not str(k).endswith('_text')) and isinstance(v, str) and v and k not in {
                "id", "task_name", "intent", "sample_id", "edit_type", "raw_instruction"
            }
        ]
        if candidates:
            output_field = candidates[-1]


    inferred_inputs = [
        k for k, v in source_sample.items()
        if (not str(k).endswith('_text')) and k != output_field and isinstance(v, str) and v
    ]
    input_fields = inferred_inputs if inferred_inputs else parsed_input_fields

    return input_fields, output_field


def extract_text_fields(sample: Dict, input_fields: List[str]) -> str:
    """Public release documentation."""
    texts = []
    for field in input_fields:
        text_field = f"{field}_text"
        if text_field in sample and sample[text_field]:
            text = sample[text_field].strip()
            if text:
                texts.append(text)

    return "\n".join(texts) if texts else ""


def process_visualcloze_g_icl_sample(args) -> Dict:
    """Public release documentation."""
    item, image_base_path, shots, weights, pre_sampled = args

    query = item.get('query', {})
    available_demos = item.get('demo', item.get('demos', []))
    if not isinstance(available_demos, list):
        available_demos = []
    if not isinstance(query, dict):
        query = {}

    input_fields, output_field = infer_visualcloze_g_fields(item)


    if not input_fields or not output_field:
        return None


    query_sources = []
    for field in input_fields:
        if field in query:
            raw_path = query[field]
            img_path = resolve_path(raw_path, image_base_path)
            query_sources.append(img_path)

    raw_query_target = query.get(output_field, '')
    query_target = resolve_path(raw_query_target, image_base_path)

    if not query_sources or not query_target:
        return None

    num_inputs = len(query_sources)



    num_demos = weighted_choice(shots, weights)
    selected_demos = available_demos[:num_demos]

    if not selected_demos:
        return None


    image_list = []
    instruction_list = []
    for demo in selected_demos:
        demo_inputs = []
        for field in input_fields:
            if field in demo:
                raw_path = demo[field]
                demo_inputs.append(resolve_path(raw_path, image_base_path))

        raw_demo_output = demo.get(output_field, '')
        demo_output = resolve_path(raw_demo_output, image_base_path)


        if len(demo_inputs) != num_inputs or not demo_output:
            continue

        # Load images
        demo_valid = True
        bytes_list = []
        for path in demo_inputs:
            b = load_image_bytes(path)
            if not b:
                demo_valid = False
                break
            bytes_list.append(b)

        if not demo_valid:
            continue

        out_b = load_image_bytes(demo_output)
        if not out_b:
            continue


        image_list.extend(bytes_list)
        image_list.append(out_b)


        demo_text = extract_text_fields(demo, input_fields)
        instruction_list.append(demo_text)


    if len(instruction_list) == 0:
        return None

    # Query sample
    query_bytes_list = []
    for path in query_sources:
        b = load_image_bytes(path)
        if not b:
            return None
        query_bytes_list.append(b)

    query_out_b = load_image_bytes(query_target)
    if not query_out_b:
        return None

    image_list.extend(query_bytes_list)
    image_list.append(query_out_b)

    query_text = extract_text_fields(query, input_fields)
    instruction_list.append(query_text)

    return {
        'image_list': image_list,
        'instruction_list': instruction_list,
        'num_inputs': num_inputs
    }


def parse_list_arg(arg_str: str) -> List[int]:
    """Public release documentation."""
    return [int(x.strip()) for x in arg_str.split(',')]


def normalize_generation_task_type(task_type: str) -> str:
    """Normalize public generation task names while keeping legacy aliases working."""
    normalized = str(task_type).strip().lower().replace("-", "_")
    aliases = {
        "t2i": "instructional_generation",
        "text2image": "instructional_generation",
        "i2i": "image_manipulation",
        "editing": "image_manipulation",
        "image_editing": "image_manipulation",
        "perfection": "visual_refinement",
        "visualcloze_g": "analogical_editing",
        "visualcloze-g": "analogical_editing",
        "fci": "fast_concept_generation",
    }
    normalized = aliases.get(normalized, normalized)
    valid = {
        "instructional_generation",
        "image_manipulation",
        "visual_refinement",
        "analogical_editing",
        "fast_concept_generation",
    }
    if normalized not in valid:
        raise ValueError(
            f"Unknown --type {task_type}. Expected one of: {', '.join(sorted(valid))}"
        )
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Convert ICL data to Parquet format")
    parser.add_argument("--input", type=str, required=True, help="Input JSON/JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        help="Generation task type: instructional_generation, image_manipulation, analogical_editing, or fast_concept_generation",
    )
    parser.add_argument(
        "--image_base_path",
        type=str,
        default=DEFAULT_IMAGE_BASE_PATH,
        help="UniICL-760K root used to resolve relative image paths. Defaults to the current UniICL-760K directory.",
    )
    parser.add_argument("--rows_per_file", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--shots", type=str, default="1,2", help="Comma-separated shot numbers")
    parser.add_argument("--weights", type=str, default="30,70", help="Comma-separated weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process (random sampling)")
    parser.add_argument("--pre_sampled", action="store_true", help="Data already has pre-sampled demos (use all available demos)")

    args = parser.parse_args()
    args.type = normalize_generation_task_type(args.type)
    random.seed(args.seed)

    shots = parse_list_arg(args.shots)
    weights = parse_list_arg(args.weights)


    if args.pre_sampled:
        shots = [99]
        weights = [100]

    if len(shots) != len(weights):
        raise ValueError("shots and weights must have same length")

    print(f"Loading data from {args.input}...")
    data = []

    # Analogical Editing uses a standard JSON list, while the other tasks use JSONL.
    if args.type == "analogical_editing":
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    print(f"Loaded {len(data)} samples.")

    # Random sampling if max_samples specified
    if args.max_samples and args.max_samples < len(data):
        data = random.sample(data, args.max_samples)
        print(f"Sampled {len(data)} samples (seed={args.seed}).")

    print(f"Processing with {args.num_workers} workers...")
    print(f"Shots: {shots}, Weights: {weights}")
    if args.image_base_path:
        print(f"Using Image Base Path: {args.image_base_path} (Overwriting absolute paths)")

    os.makedirs(args.output, exist_ok=True)

    process_func = None
    if args.type == "instructional_generation":
        process_func = process_t2i_icl_sample
    elif args.type in {"image_manipulation", "visual_refinement"}:
        process_func = process_i2i_icl_sample
    elif args.type == "analogical_editing":
        process_func = process_visualcloze_g_icl_sample
    elif args.type == "fast_concept_generation":
        process_func = process_fci_icl_sample

    valid_samples = []
    
    # Prepare args for multiprocessing map
    map_args = [(item, args.image_base_path, shots, weights, args.pre_sampled) for item in data]

    if args.num_workers <= 1:
        for map_arg in tqdm(map_args, total=len(map_args), desc="Processing"):
            result = process_func(map_arg)
            if result:
                valid_samples.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for result in tqdm(executor.map(process_func, map_args), total=len(map_args), desc="Processing"):
                if result:
                    valid_samples.append(result)

    print(f"Generated {len(valid_samples)} valid samples.")

    # Save to parquet files
    schema = None
    if args.type == "analogical_editing":
        schema = pa.schema([
            ('image_list', pa.list_(pa.binary())),
            ('instruction_list', pa.list_(pa.string())),
            ('num_inputs', pa.int32())
        ])
    else:
        schema = pa.schema([
            ('image_list', pa.list_(pa.binary())),
            ('instruction_list', pa.list_(pa.string()))
        ])

    num_files = (len(valid_samples) + args.rows_per_file - 1) // args.rows_per_file

    print(f"Saving to {num_files} parquet files in {args.output}...")

    parquet_info = {}

    for i in range(num_files):
        start_idx = i * args.rows_per_file
        end_idx = min((i + 1) * args.rows_per_file, len(valid_samples))
        batch = valid_samples[start_idx:end_idx]

        # Transpose list of dicts to dict of lists
        table_data = defaultdict(list)
        for item in batch:
            for k, v in item.items():
                table_data[k].append(v)

        table = pa.Table.from_pydict(table_data, schema=schema)
        output_file = os.path.join(args.output, f"part_{i:05d}.parquet")
        pq.write_table(table, output_file)

        # Record parquet info
        parquet_info[output_file] = {
            'num_row_groups': 1,
            'num_rows': len(batch)
        }

    # Write parquet_info.json
    info_path = os.path.join(args.output, "parquet_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(parquet_info, f, indent=2)

    print(f"Written parquet_info.json with {len(parquet_info)} entries.")
    print("Done.")

if __name__ == "__main__":
    main()
