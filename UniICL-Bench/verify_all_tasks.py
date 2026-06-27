#!/usr/bin/env python3
"""Public release module documentation."""

import argparse
import json
import os
from pathlib import Path

from public_path_config import BENCHMARK_ROOT, TASK_DATA_REL_PATHS, get_task_image_dir


TASK_DISPLAY_NAMES = {
    "visual_grounding": "Visual Grounding",
    "attribute_recognition": "Attribute Recognition",
    "scene_reasoning": "Scene Reasoning",
    "style_aware_caption": "Style-Aware Caption",
    "instructional_generation": "Instructional Generation",
    "image_manipulation": "Image Manipulation",
    "visual_refinement": "Visual Refinement",
    "analogical_editing": "Analogical Editing",
    "aesthetic_assessment": "Aesthetic Assessment",
    "forgery_detection": "Forgery Detection",
    "fast_concept_mapping": "Fast Concept Mapping",
    "fast_concept_generation": "Fast Concept Generation",
    "world_aware_planning": "World-Aware Planning",
    "chain_of_editing": "Chain-of-Editing",
    "analogical_inference": "Analogical Inference",
}


TASK_CONFIG = {
    # Understanding Tasks - LAION-HR
    "visual_grounding": {
        "data_path": TASK_DATA_REL_PATHS["visual_grounding"],
        "image_dir": get_task_image_dir("visual_grounding", BENCHMARK_ROOT),
        "has_gt": True,
        "image_fields": ["image_name"],
        "gt_fields": ["answer", "annotation"],
    },
    "attribute_recognition": {
        "data_path": TASK_DATA_REL_PATHS["attribute_recognition"],
        "image_dir": get_task_image_dir("attribute_recognition", BENCHMARK_ROOT),
        "has_gt": True,
        "image_fields": ["image_name"],
        "gt_fields": ["answer", "annotation"],
    },
    "scene_reasoning": {
        "data_path": TASK_DATA_REL_PATHS["scene_reasoning"],
        "image_dir": get_task_image_dir("scene_reasoning", BENCHMARK_ROOT),
        "has_gt": True,
        "image_fields": ["image_name"],
        "gt_fields": ["answer", "annotation"],
    },
    "style_aware_caption": {
        "data_path": TASK_DATA_REL_PATHS["style_aware_caption"],
        "image_dir": get_task_image_dir("style_aware_caption", BENCHMARK_ROOT),
        "has_gt": True,
        "image_fields": ["image_name"],
        "gt_fields": ["answer", "annotation"],
    },
    "analogical_inference": {
        "data_path": TASK_DATA_REL_PATHS["analogical_inference"],
        "image_dir": get_task_image_dir("analogical_inference", BENCHMARK_ROOT),
        "has_gt": True,
        "image_fields": ["image_name"],
        "gt_fields": ["answer"],
    },


    "instructional_generation": {
        "data_path": TASK_DATA_REL_PATHS["instructional_generation"],
        "image_dir": get_task_image_dir("instructional_generation", BENCHMARK_ROOT),
        "has_gt": True,
        "image_fields": ["image_name", "answer"],
        "gt_fields": [],
    },
    "image_manipulation": {
        "data_path": TASK_DATA_REL_PATHS["image_manipulation"],
        "image_dir": get_task_image_dir("image_manipulation", BENCHMARK_ROOT),
        "has_gt": True,
        "image_fields": ["image_name"],
        "gt_fields": ["answer"],
    },
    "visual_refinement": {
        "data_path": TASK_DATA_REL_PATHS["visual_refinement"],
        "image_dir": get_task_image_dir("visual_refinement", BENCHMARK_ROOT),
        "has_gt": True,
        "image_fields": ["image_name"],
        "gt_fields": ["answer"],
    },
    "analogical_editing": {
        "data_path": TASK_DATA_REL_PATHS["analogical_editing"],
        "image_dir": get_task_image_dir("analogical_editing", BENCHMARK_ROOT),
        "has_gt": True,
        "image_fields": [],
        "gt_fields": [],
        "special": "visualcloze",
    },

    # Assessment Tasks
    "aesthetic_assessment": {
        "data_path": TASK_DATA_REL_PATHS["aesthetic_assessment"],
        "image_dir": get_task_image_dir("aesthetic_assessment", BENCHMARK_ROOT),
        "has_gt": True,
        "image_fields": ["image_name"],
        "gt_fields": ["answer", "annotation"],
    },
    "forgery_detection": {
        "data_path": TASK_DATA_REL_PATHS["forgery_detection"],
        "image_dir": get_task_image_dir("forgery_detection", BENCHMARK_ROOT),
        "has_gt": True,
        "image_fields": ["image_name"],
        "gt_fields": ["answer", "annotation"],
    },

    # Special Tasks
    "fast_concept_mapping": {
        "data_path": TASK_DATA_REL_PATHS["fast_concept_mapping"],
        "image_dir": get_task_image_dir("fast_concept_mapping", BENCHMARK_ROOT),
        "has_gt": True,
        "image_fields": [],  # query.image
        "gt_fields": [],
        "special": "fcb",
    },
    "fast_concept_generation": {
        "data_path": TASK_DATA_REL_PATHS["fast_concept_generation"],
        "image_dir": get_task_image_dir("fast_concept_generation", BENCHMARK_ROOT),
        "has_gt": True,
        "image_fields": [],  # query.image
        "gt_fields": [],
        "special": "fci",
    },
    "world_aware_planning": {
        "data_path": TASK_DATA_REL_PATHS["world_aware_planning"],
        "image_dir": get_task_image_dir("world_aware_planning", BENCHMARK_ROOT),
        "has_gt": True,
        "image_fields": [],
        "gt_fields": [],
        "special": "planning",
    },
    "chain_of_editing": {
        "data_path": TASK_DATA_REL_PATHS["chain_of_editing"],
        "image_dir": get_task_image_dir("chain_of_editing", BENCHMARK_ROOT),
        "has_gt": True,
        "image_fields": [],
        "gt_fields": [],
        "special": "chain_edit",
    },
}


def load_data(data_path):
    """Public release documentation."""
    ext = os.path.splitext(data_path)[1]
    with open(data_path, 'r') as f:
        if ext == '.jsonl':
            return [json.loads(line) for line in f]
        else:  # .json
            data = json.load(f)

            if isinstance(data, dict):

                if 'samples' in data:
                    return data['samples']

                elif 'data' in data:
                    return data['data']
            return data


def check_image_path(image_dir, relative_path, desc=""):
    """Public release documentation."""
    if not relative_path:
        return False, f"Empty path: {desc}"

    full_path = os.path.join(image_dir, relative_path)
    if not os.path.exists(full_path):
        return False, f"Missing: {full_path}"

    return True, full_path


def verify_standard_task(task_name, config, benchmark_dir, check_all=False, max_samples=5):
    """Public release documentation."""
    task_display_name = TASK_DISPLAY_NAMES.get(task_name, task_name.replace("_", " ").title())
    print(f"\n{'='*80}")
    print(f"Verifying task: {task_display_name}")
    print(f"{'='*80}")

    data_path = os.path.join(benchmark_dir, config['data_path'])
    image_dir = config['image_dir']

    print(f"Data file: {data_path}")
    print(f"Image dir: {image_dir}")

    if not os.path.exists(data_path):
        print(f"ERROR: data file not found: {data_path}")
        return {"total": 0, "missing": 0, "errors": [f"Data file not found: {data_path}"]}

    data = load_data(data_path)
    print(f"Sample count: {len(data)}")


    samples_to_check = data if check_all else data[:max_samples]
    print(f"Checked samples: {len(samples_to_check)}")

    stats = {
        "total_images": 0,
        "missing_images": 0,
        "errors": [],
    }

    for idx, item in enumerate(samples_to_check):
        sample_id = item.get('id', item.get('image_name', f'sample_{idx}'))


        for field in config['image_fields']:
            if field in item:
                stats["total_images"] += 1
                ok, msg = check_image_path(image_dir, item[field], f"{sample_id}.{field}")
                if not ok:
                    stats["missing_images"] += 1
                    stats["errors"].append(msg)


        for field in config['gt_fields']:
            if field in item:

                gt_value = item[field]
                if isinstance(gt_value, str) and any(ext in gt_value for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                    stats["total_images"] += 1
                    ok, msg = check_image_path(image_dir, gt_value, f"{sample_id}.{field}(GT)")
                    if not ok:
                        stats["missing_images"] += 1
                        stats["errors"].append(msg)


        demos = item.get('demos', [])
        for demo_idx, demo in enumerate(demos[:3]):

            for field in config['image_fields']:
                if field in demo:
                    stats["total_images"] += 1
                    ok, msg = check_image_path(image_dir, demo[field], f"{sample_id}.demo{demo_idx}.{field}")
                    if not ok:
                        stats["missing_images"] += 1
                        stats["errors"].append(msg)


            for field in config['gt_fields']:
                if field in demo:
                    gt_value = demo[field]
                    if isinstance(gt_value, str) and any(ext in gt_value for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                        stats["total_images"] += 1
                        ok, msg = check_image_path(image_dir, gt_value, f"{sample_id}.demo{demo_idx}.{field}(GT)")
                        if not ok:
                            stats["missing_images"] += 1
                            stats["errors"].append(msg)


    print(f"\nVerification summary:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Missing images: {stats['missing_images']}")

    if stats['missing_images'] > 0:
        print(f"\nERROR: missing images (first 10):")
        for err in stats['errors'][:10]:
            print(f"  {err}")
    else:
        print(f"\nAll image paths are valid.")

    return stats


def verify_visualcloze_g(task_name, config, benchmark_dir, check_all=False, max_samples=5):
    """Public release documentation."""
    task_display_name = TASK_DISPLAY_NAMES.get(task_name, task_name.replace("_", " ").title())
    print(f"\n{'='*80}")
    print(f"Verifying task: {task_display_name} (special format)")
    print(f"{'='*80}")

    data_path = os.path.join(benchmark_dir, config['data_path'])
    image_dir = config['image_dir']

    print(f"Data file: {data_path}")
    print(f"Image dir: {image_dir}")

    if not os.path.exists(data_path):
        print(f"ERROR: data file not found: {data_path}")
        return {"total": 0, "missing": 0, "errors": [f"Data file not found: {data_path}"]}

    data = load_data(data_path)
    print(f"Sample count: {len(data)}")

    samples_to_check = data if check_all else data[:max_samples]
    print(f"Checked samples: {len(samples_to_check)}")

    stats = {
        "total_images": 0,
        "missing_images": 0,
        "errors": [],
    }

    for idx, item in enumerate(samples_to_check):
        item_id = item.get('id', f'item_{idx}')



        query = item.get('query', {})


        for key, value in query.items():
            if isinstance(value, str) and any(ext in value for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                stats["total_images"] += 1
                ok, msg = check_image_path(image_dir, value, f"{item_id}.query.{key}")
                if not ok:
                    stats["missing_images"] += 1
                    stats["errors"].append(msg)


        demos = item.get('demos', [])
        for demo_idx, demo in enumerate(demos[:3]):
            for key, value in demo.items():
                if isinstance(value, str) and any(ext in value for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                    stats["total_images"] += 1
                    ok, msg = check_image_path(image_dir, value, f"{item_id}.demo{demo_idx}.{key}")
                    if not ok:
                        stats["missing_images"] += 1
                        stats["errors"].append(msg)

    print(f"\nVerification summary:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Missing images: {stats['missing_images']}")

    if stats['missing_images'] > 0:
        print(f"\nERROR: missing images (first 10):")
        for err in stats['errors'][:10]:
            print(f"  {err}")
    else:
        print(f"\nAll image paths are valid.")

    return stats


def verify_fcb_fci(task_name, config, benchmark_dir, check_all=False, max_samples=5):
    """Public release documentation."""
    task_display_name = TASK_DISPLAY_NAMES.get(task_name, task_name.replace("_", " ").title())
    print(f"\n{'='*80}")
    print(f"Verifying task: {task_display_name} (special format)")
    print(f"{'='*80}")

    data_path = os.path.join(benchmark_dir, config['data_path'])
    image_dir = config['image_dir']

    print(f"Data file: {data_path}")
    print(f"Image dir: {image_dir}")

    if not os.path.exists(data_path):
        print(f"ERROR: data file not found: {data_path}")
        return {"total": 0, "missing": 0, "errors": [f"Data file not found: {data_path}"]}

    data = load_data(data_path)
    print(f"Sample count: {len(data)}")

    samples_to_check = data if check_all else data[:max_samples]
    print(f"Checked samples: {len(samples_to_check)}")

    stats = {
        "total_images": 0,
        "missing_images": 0,
        "errors": [],
    }

    for idx, item in enumerate(samples_to_check):

        query = item.get('query', {})
        query_img = query.get('image', '')

        if query_img:
            stats["total_images"] += 1
            ok, msg = check_image_path(image_dir, query_img, f"item{idx}.query.image")
            if not ok:
                stats["missing_images"] += 1
                stats["errors"].append(msg)


        demos = item.get('demos', [])
        for demo_idx, demo in enumerate(demos[:3]):
            demo_img = demo.get('image', '')
            if demo_img:
                stats["total_images"] += 1
                ok, msg = check_image_path(image_dir, demo_img, f"item{idx}.demo{demo_idx}.image")
                if not ok:
                    stats["missing_images"] += 1
                    stats["errors"].append(msg)

    print(f"\nVerification summary:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Missing images: {stats['missing_images']}")

    if stats['missing_images'] > 0:
        print(f"\nERROR: missing images (first 10):")
        for err in stats['errors'][:10]:
            print(f"  {err}")
    else:
        print(f"\nAll image paths are valid.")

    return stats


def verify_planning(task_name, config, benchmark_dir, check_all=False, max_samples=5):
    """Public release documentation."""
    task_display_name = TASK_DISPLAY_NAMES.get(task_name, task_name.replace("_", " ").title())
    print(f"\n{'='*80}")
    print(f"Verifying task: {task_display_name} (special format)")
    print(f"{'='*80}")

    data_path = os.path.join(benchmark_dir, config['data_path'])
    image_dir = config['image_dir']

    print(f"Data file: {data_path}")
    print(f"Image dir: {image_dir}")

    if not os.path.exists(data_path):
        print(f"ERROR: data file not found: {data_path}")
        return {"total_images": 0, "missing_images": 0, "errors": [f"Data file not found: {data_path}"]}

    data = load_data(data_path)
    print(f"Sample count: {len(data)}")

    samples_to_check = data if check_all else data[:max_samples]
    print(f"Checked samples: {len(samples_to_check)}")

    stats = {
        "total_images": 0,
        "missing_images": 0,
        "errors": [],
    }

    for idx, item in enumerate(samples_to_check):
        sample_id = item.get('id', f'sample_{idx}')


        images = item.get('images', [])
        for img_idx, img_path in enumerate(images):
            stats["total_images"] += 1
            ok, msg = check_image_path(image_dir, img_path, f"{sample_id}.images[{img_idx}]")
            if not ok:
                stats["missing_images"] += 1
                stats["errors"].append(msg)

    print(f"\nVerification summary:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Missing images: {stats['missing_images']}")

    if stats['missing_images'] > 0:
        print(f"\nERROR: missing images (first 10):")
        for err in stats['errors'][:10]:
            print(f"  {err}")
    else:
        print(f"\nAll image paths are valid.")

    return stats


def verify_chain_edit(task_name, config, benchmark_dir, check_all=False, max_samples=5):
    """Public release documentation."""
    task_display_name = TASK_DISPLAY_NAMES.get(task_name, task_name.replace("_", " ").title())
    print(f"\n{'='*80}")
    print(f"Verifying task: {task_display_name} (special format)")
    print(f"{'='*80}")

    data_path = os.path.join(benchmark_dir, config['data_path'])
    image_dir = config['image_dir']

    print(f"Data file: {data_path}")
    print(f"Image dir: {image_dir}")

    if not os.path.exists(data_path):
        print(f"ERROR: data file not found: {data_path}")
        return {"total_images": 0, "missing_images": 0, "errors": [f"Data file not found: {data_path}"]}

    data = load_data(data_path)
    print(f"Sample count: {len(data)}")

    samples_to_check = data if check_all else data[:max_samples]
    print(f"Checked samples: {len(samples_to_check)}")

    stats = {
        "total_images": 0,
        "missing_images": 0,
        "errors": [],
    }

    for idx, item in enumerate(samples_to_check):
        sample_id = item.get('id', f'sample_{idx}')


        original_img = item.get('original_image', '')
        if original_img:
            stats["total_images"] += 1
            ok, msg = check_image_path(image_dir, original_img, f"{sample_id}.original_image")
            if not ok:
                stats["missing_images"] += 1
                stats["errors"].append(msg)


        edit_steps = item.get('edit_steps', [])
        for step_idx, step in enumerate(edit_steps):
            ref_img = step.get('reference_image', '')
            if ref_img:
                stats["total_images"] += 1
                ok, msg = check_image_path(image_dir, ref_img, f"{sample_id}.edit_steps[{step_idx}].reference_image")
                if not ok:
                    stats["missing_images"] += 1
                    stats["errors"].append(msg)

    print(f"\nVerification summary:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Missing images: {stats['missing_images']}")

    if stats['missing_images'] > 0:
        print(f"\nERROR: missing images (first 10):")
        for err in stats['errors'][:10]:
            print(f"  {err}")
    else:
        print(f"\nAll image paths are valid.")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Verify all benchmark image references.")
    parser.add_argument("--benchmark-dir", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--check-all", action="store_true", help="Check all samples instead of a small prefix.")
    parser.add_argument("--max-samples", type=int, default=5, help="Prefix length when --check-all is not set.")
    args = parser.parse_args()

    benchmark_dir = str(args.benchmark_dir)
    check_all = args.check_all

    print("="*80)
    print("UniICL-Bench image-path verification across all tasks")
    print("="*80)
    print(f"\nUniICL-Bench dir: {os.path.abspath(benchmark_dir)}")
    print(f"Verification mode: {'all samples' if check_all else 'first 5 samples'}")
    print(f"\nNote: local runs may not be able to access image files stored on a remote server")
    print("=" * 80)

    all_stats = {}

    for task_name, config in TASK_CONFIG.items():
        try:
            special_type = config.get('special')
            if special_type == 'visualcloze':
                stats = verify_visualcloze_g(task_name, config, benchmark_dir, check_all, args.max_samples)
            elif special_type in ['fcb', 'fci']:
                stats = verify_fcb_fci(task_name, config, benchmark_dir, check_all, args.max_samples)
            elif special_type == 'planning':
                stats = verify_planning(task_name, config, benchmark_dir, check_all, args.max_samples)
            elif special_type == 'chain_edit':
                stats = verify_chain_edit(task_name, config, benchmark_dir, check_all, args.max_samples)
            else:
                stats = verify_standard_task(task_name, config, benchmark_dir, check_all, args.max_samples)

            all_stats[task_name] = stats
        except Exception as e:
            print(f"\nERROR: task {task_name} verification failed: {e}")
            import traceback
            traceback.print_exc()
            all_stats[task_name] = {"total_images": 0, "missing_images": 0, "errors": [str(e)]}


    print("\n" + "="*80)
    print("Overall verification summary")
    print("="*80)

    total_tasks = len(all_stats)
    total_images = sum(s.get("total_images", 0) for s in all_stats.values())
    total_missing = sum(s.get("missing_images", 0) for s in all_stats.values())

    print(f"\nTotal tasks: {total_tasks}")
    print(f"Total checked images: {total_images}")
    print(f"Total missing images: {total_missing}")

    print(f"\nPer-task summary:")
    for task_name, stats in all_stats.items():
        total = stats.get("total_images", 0)
        missing = stats.get("missing_images", 0)
        status = "✅" if missing == 0 and total > 0 else "❌"
        task_display_name = TASK_DISPLAY_NAMES.get(task_name, task_name.replace("_", " ").title())
        print(f"  {status} {task_display_name:24s}: {total:4d} images, {missing:4d} missing")

    if total_missing == 0 and total_images > 0:
        print(f"\nAll benchmark image paths are valid.")
    elif total_images == 0:
        print(f"\nWarning: no images were checked (possible data-path issue).")
    else:
        print(f"\nFound {total_missing}/{total_images} missing images")
        print(f"\nTip: this can be expected on a local machine if the images live on a remote server.")
        print(f"     Run this script on the target server to validate the real paths.")

    print("="*80)


if __name__ == '__main__':
    main()
