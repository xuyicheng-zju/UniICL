#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Public release module documentation."""

import json
import os
import re
import random
import string
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from tqdm import tqdm
import argparse


DEFAULT_IMAGE_BASE_PATH = str(Path(__file__).resolve().parent)

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


def get_task_type(task_name: str) -> str:
    """Public release documentation."""
    task_name_lower = task_name.lower()
    if task_name_lower in [
        'stylized_caption', 'style_aware_caption',
        'vqa', 'scene_reasoning',
        'grounding', 'visual_grounding',
        'attr_rec', 'attribute_recognition',
        'aigi', 'forgery_detection',
        'ava', 'aesthetic_assessment',
        'fcb', 'fast_concept_mapping',
        'planning', 'world_aware_planning',
        'visualcloze-u', 'visualcloze_u', 'analogical_inference',
        'visualdialog', 'visdial',
    ]:
        return 'UND'
    elif task_name_lower in ['t2i', 'text2image', 'instructional_generation']:
        return 'GEN'
    elif task_name_lower in [
        'i2i', 'editing', 'image_editing',
        'image_manipulation', 'visual_refinement',
        'visualcloze-g', 'visualcloze_g', 'analogical_editing',
        'fci', 'fast_concept_generation',
    ]:
        return 'EDIT'
    else:
        return 'AUTO'


def detect_task_type_from_sample(sample: Dict) -> str:
    """Public release documentation."""
    answer = sample.get('answer', '') or sample.get('annotation', '')


    if isinstance(answer, str) and (
        answer.endswith('.png') or
        answer.endswith('.jpg') or
        answer.endswith('.jpeg') or
        ('/' in answer and not answer.startswith('<'))
    ):
        if 'image_path' in sample or 'image_name' in sample:
            demos = sample.get('demos', [])
            if demos and 'answer' in demos[0]:
                demo_answer = demos[0].get('answer', '') or demos[0].get('annotation', '')
                if isinstance(demo_answer, str) and ('/' in demo_answer or demo_answer.endswith('.png')):
                    return 'EDIT'
        return 'GEN'

    return 'UND'


def resolve_image_path(item: Dict, keys: List[str], base_path: str = "") -> str:
    """Public release documentation."""
    val = None
    for k in keys:
        if item.get(k):
            val = item[k]
            break

    if not val:
        return ""

    if base_path and val.startswith('/'):
        filename = os.path.basename(val)
        return os.path.join(base_path, filename)

    return val








def convert_und_task(sample: Dict, image_base_path: str = "", num_demos: int = 2) -> Dict:
    """Public release documentation."""
    available_demos = sample.get('demos', [])
    

    demos = available_demos[:num_demos]

    # Standard UND
    query_image = resolve_image_path(sample, ['image_name', 'image_path'], image_base_path)
    query_text = sample.get('text') or sample.get('instruction', '')
    query_answer = sample.get('answer') or sample.get('annotation', '')

    # Fallback if no text
    if not query_text and 'question' in sample:
        query_text = sample['question']

    images = []
    conversations = []
    human_parts = []

    for demo in demos:
        demo_image = resolve_image_path(demo, ['image_path', 'image_name'], image_base_path)
        demo_text = demo.get('instruction') or demo.get('text', '')
        demo_answer = demo.get('answer') or demo.get('annotation', '')

        images.append(demo_image)
        # Demo: User: <image>\n{text}\nAssistant: {answer}
        human_parts.append(f"User: <image>\n{demo_text}\nAssistant: {demo_answer}")

    images.append(query_image)
    # Query: User: <image>\n{text}\nAssistant:
    human_parts.append(f"User: <image>\n{query_text}\nAssistant:")

    conversations.append({
        "from": "human",
        "value": "\n".join(human_parts)
    })


    conversations.append({
        "from": "gpt",
        "value": query_answer
    })

    return {
        "image": images,
        "conversations": conversations
    }


def convert_planning_task(sample: Dict, image_base_path: str = "") -> Dict:
    """Public release documentation."""
    raw_images = sample.get('images', [])
    resolved_images = []

    for img_path in raw_images:
        path = resolve_image_path({'p': img_path}, ['p'], image_base_path)
        if path:
            resolved_images.append(path)

    conversations = sample.get('conversations', [])



    if conversations and conversations[-1].get('from') in ['gpt', 'Assistant']:
        conversations = conversations[:-1]

    if conversations and conversations[-1].get('from') == 'observation':
        conversations = conversations[:-1]


    if len(raw_images) > 10:
        return None

    if len(conversations) < 2:
        return None


    context_parts = []
    pairs = []  # [(observation_value, gpt_value), ...]
    current_obs = None
    task_desc = None
    image_count = 0

    for conv in conversations:
        role = conv.get('from', '')
        value = conv.get('value', '')

        if role == 'human':
            task_desc = value
        elif role == 'observation':
            current_obs = value  # <image>
            image_count += 1
        elif role in ['gpt', 'Assistant']:
            if value.strip().lower() == 'sure!':
                continue
            if current_obs is not None:
                pairs.append((current_obs, value))
                current_obs = None
            else:
                # gpt without preceding observation (shouldn't happen normally)
                pairs.append((None, value))

    if not pairs or task_desc is None:
        return None


    context_parts.append(task_desc)

    for i, (obs, gpt_val) in enumerate(pairs):
        if obs:
            context_parts.append(f"Observation: {obs}")  # Observation: <image>
        if i < len(pairs) - 1:

            context_parts.append(f"Assistant: {gpt_val}")


    last_gpt_value = pairs[-1][1]


    used_images = resolved_images[:image_count] if resolved_images else []


    human_value = "\n".join(context_parts) + "\nAssistant:"

    return {
        "image": used_images,
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": last_gpt_value}
        ]
    }


def convert_visdial_task(sample: Dict, image_base_path: str = "") -> Dict:
    """Public release documentation."""
    image_id = sample.get('image_id', '')
    caption = sample.get('caption', '')
    dialog = sample.get('dialog', [])

    if not dialog or len(dialog) < 2:
        return None


    if image_base_path:
        if isinstance(image_id, int):
            image_filename = f"COCO_train2014_{image_id:012d}.jpg"
        else:
            image_filename = f"COCO_train2014_{int(image_id):012d}.jpg"
        query_image = os.path.join(image_base_path, image_filename)
    else:
        query_image = str(image_id)


    query_turn = dialog[-1]
    query_question = query_turn.get('question', '')
    query_answer = query_turn.get('answer', '')


    history_turns = dialog[:-1]


    context_parts = []
    if caption:
        context_parts.append(f"Image description: {caption}")

    for turn in history_turns:
        context_parts.append(f"User: {turn['question']}\nAssistant: {turn['answer']}")


    dialog_context = "\n".join(context_parts) if context_parts else ""

    if dialog_context:
        prompt = f"<image>\n{dialog_context}\nUser: {query_question}\nAssistant:"
    else:
        prompt = f"<image>\nUser: {query_question}\nAssistant:"

    images = [query_image]
    conversations = [
        {
            "from": "human",
            "value": prompt
        },
        {
            "from": "gpt",
            "value": query_answer
        }
    ]

    return {
        "image": images,
        "conversations": conversations
    }


def convert_sample(sample: Dict, task_type: str, image_base_path: str, num_demos: int) -> Dict:
    """Public release documentation."""
    # Normalize sample: flatten the query dict if present (for example, Fast Concept Mapping)
    if 'query' in sample and isinstance(sample['query'], dict):
        query = sample['query']
        for k, v in query.items():
            if k not in sample:
                sample[k] = v

    if task_type == "AUTO":
        sample_task = sample.get('task_type', '')
        if sample_task:
            task_type = get_task_type(sample_task)
        if task_type == "AUTO":
            task_type = detect_task_type_from_sample(sample)

    if task_type == "UND":
        return convert_und_task(sample, image_base_path, num_demos)
    elif task_type == "PLANNING":
        return convert_planning_task(sample, image_base_path)
    elif task_type == "VISDIAL":
        return convert_visdial_task(sample, image_base_path)
    else:
        raise ValueError(f"Unknown task type: {task_type}. GEN/EDIT tasks should use convert_icl_to_parquet.py")

def convert_dataset(
    input_jsonl: str,
    output_jsonl: str,
    task_type: str = "AUTO",
    image_base_path: str = "",
    shots: List[int] = [1, 2],
    weights: List[int] = [30, 70],
    max_samples: Optional[int] = None,
    seed: int = 42
):
    """Public release documentation."""
    random.seed(seed)

    print(f"Loading: {input_jsonl}")
    print(f"Task type: {task_type}")
    print(f"Shots: {shots}, Weights: {weights}")

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0
    shot_dist = {}

    # Detect whether the input uses a JSON list (World-Aware Planning) or JSONL.
    is_json_list = False
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        # Check first non-whitespace character
        chunk = f.read(1024).strip()
        if chunk.startswith('['):
            is_json_list = True

    # Load all data first for random sampling
    print("Loading data...")
    all_data = []
    if is_json_list:
        with open(input_jsonl, 'r', encoding='utf-8') as fin:
            all_data = json.load(fin)
    else:
        with open(input_jsonl, 'r', encoding='utf-8') as fin:
            for line in fin:
                if line.strip():
                    all_data.append(json.loads(line))

    print(f"Loaded {len(all_data)} samples.")

    # Random sampling if max_samples specified
    if max_samples and max_samples < len(all_data):
        all_data = random.sample(all_data, max_samples)
        print(f"Sampled {len(all_data)} samples (seed={seed}).")

    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for sample in tqdm(all_data, desc="Converting"):
            try:

                available_demos = len(sample.get('demos', []))
                actual_demos = 0
                if available_demos > 0:
                    num_demos = weighted_choice(shots, weights)
                    actual_demos = min(num_demos, available_demos)


                uniicl_item = convert_sample(sample, task_type, image_base_path, actual_demos)


                shot_dist[actual_demos] = shot_dist.get(actual_demos, 0) + 1

                if uniicl_item:
                    fout.write(json.dumps(uniicl_item, ensure_ascii=False) + '\n')
                    processed_count += 1
                else:
                    skipped_count += 1

            except Exception as e:
                # print(f"Error processing sample: {e}")
                skipped_count += 1
                continue

    print(f"\nConversion completed!")
    print(f"  Processed: {processed_count:,}")
    print(f"  Skipped: {skipped_count:,}")
    print(f"  Shot distribution: {shot_dist}")
    print(f"  Output: {output_jsonl}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert ICL JSONL to UniICL Format")
    

    parser.add_argument('--input_jsonl', type=str, required=True)
    parser.add_argument('--output_jsonl', type=str, required=True)
    parser.add_argument('--task_type', type=str, default='AUTO')
    parser.add_argument(
        '--image_base_path',
        type=str,
        default=DEFAULT_IMAGE_BASE_PATH,
        help="UniICL-760K root used to resolve relative image paths. Defaults to the current UniICL-760K directory.",
    )
    parser.add_argument('--shots', type=str, default='1,2')
    parser.add_argument('--weights', type=str, default='30,70')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--pre_sampled', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_history', type=int, default=1)
    parser.add_argument('--max_history', type=int, default=10)
    parser.add_argument('--max_turns_per_dialog', type=int, default=3)

    args = parser.parse_args()

    shots_list = [int(s) for s in args.shots.split(',')]
    weights_list = [int(w) for w in args.weights.split(',')]

    convert_dataset(
        args.input_jsonl, args.output_jsonl,
        task_type=args.task_type, image_base_path=args.image_base_path,
        shots=shots_list, weights=weights_list,
        max_samples=args.max_samples, seed=args.seed
    )
