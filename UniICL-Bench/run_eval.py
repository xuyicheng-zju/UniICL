
"""Public release module documentation."""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from public_path_config import (
    BENCHMARK_ROOT,
    CANONICAL_TASK_ORDER,
    DEFAULT_FLUX_MODEL,
    DEFAULT_HPSV3_CHECKPOINT,
    DEFAULT_INTERNVL_MODEL,
    DEFAULT_JUDGE_API_BASE,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_NEXUSGEN_MODEL,
    DEFAULT_OVISU1_MODEL,
    DEFAULT_QWEN25VL_MODEL,
    DEFAULT_QWEN3VL_MODEL,
    DEFAULT_SIGLIP_MODEL,
    DEFAULT_UNIICL_BASE_MODEL,
    DEFAULT_UNIICL_FINETUNED_MODEL,
    DEFAULT_UNIWORLD_MODEL,
    get_task_data_path as resolve_task_data_path,
    get_task_image_dir as resolve_task_image_dir,
    normalize_task_name,
)

# ================================================================================

# ================================================================================


OUTPUT_BASE_DIR = "./eval_results_capm_final"
LOG_BASE_DIR = "./logs_capm_final"

CAPM_ABLATION_CHOICES = [
    "none",
    "no_adaptive_routing",
    "no_decoupled_encoding",
    "no_low_rank_transformation",
]



GPUS = [0, 1, 2, 3, 4, 5, 6]


GPU_TASKS = {
    0: ["visual_grounding", "attribute_recognition"],
    1: ["scene_reasoning", "style_aware_caption"],
    2: ["instructional_generation", "image_manipulation"],
    3: ["fast_concept_mapping", "fast_concept_generation"],
    4: ["world_aware_planning", "chain_of_editing"],
    5: ["analogical_inference", "analogical_editing"],
    6: ["aesthetic_assessment", "forgery_detection", "visual_refinement"],
}

# ================================================================================

# ================================================================================

def get_task_image_dir(task: str) -> str:
    """Public release documentation."""
    return resolve_task_image_dir(task, BENCHMARK_ROOT)


def get_task_data_path(task: str, benchmark_dir: str = ".") -> str:
    """Public release documentation."""
    return resolve_task_data_path(task, benchmark_dir)


def get_task_kshot_range(task: str) -> List[int]:
    """Public release documentation."""

    if task in ["world_aware_planning", "chain_of_editing"]:
        return [0]


    return [0, 1, 2, 4, 8]


def get_model_supported_tasks(model: str) -> List[str]:
    """Public release documentation."""
    config = {
        "uniicl": [
            "visual_grounding", "attribute_recognition", "scene_reasoning", "style_aware_caption",
            "instructional_generation", "image_manipulation", "aesthetic_assessment",
            "forgery_detection", "visual_refinement", "fast_concept_mapping",
            "fast_concept_generation", "world_aware_planning", "chain_of_editing",
            "analogical_editing", "analogical_inference",
        ],
        "uniworld": [
            "visual_grounding", "attribute_recognition", "scene_reasoning", "style_aware_caption",
            "instructional_generation", "image_manipulation", "aesthetic_assessment",
            "forgery_detection", "visual_refinement", "fast_concept_mapping",
            "fast_concept_generation", "world_aware_planning", "chain_of_editing",
            "analogical_editing", "analogical_inference",
        ],
        "qwen3vl": [
            "visual_grounding", "attribute_recognition", "scene_reasoning",
            "style_aware_caption", "aesthetic_assessment", "forgery_detection",
            "fast_concept_mapping", "world_aware_planning", "analogical_inference",
        ],
        "qwen25vl": [
            "visual_grounding", "attribute_recognition", "scene_reasoning",
            "style_aware_caption", "aesthetic_assessment", "forgery_detection",
            "fast_concept_mapping", "world_aware_planning", "analogical_inference",
        ],
        "internvl": [
            "visual_grounding", "attribute_recognition", "scene_reasoning",
            "style_aware_caption", "aesthetic_assessment", "forgery_detection",
            "fast_concept_mapping", "world_aware_planning", "analogical_inference",
        ],
        "nexusgen": [
            "visual_grounding", "attribute_recognition", "scene_reasoning", "style_aware_caption",
            "instructional_generation", "image_manipulation", "aesthetic_assessment",
            "forgery_detection", "visual_refinement", "fast_concept_mapping",
            "fast_concept_generation", "world_aware_planning", "chain_of_editing",
            "analogical_editing", "analogical_inference",
        ],
        "ovisu1": [
            "visual_grounding", "attribute_recognition", "scene_reasoning", "style_aware_caption",
            "instructional_generation", "image_manipulation", "aesthetic_assessment",
            "forgery_detection", "visual_refinement", "fast_concept_mapping",
            "fast_concept_generation", "world_aware_planning", "chain_of_editing",
            "analogical_editing", "analogical_inference",
        ],
    }

    if model not in config:
        raise ValueError(f"Unknown model: {model}")

    return config[model]


def is_task_supported(model: str, task: str) -> bool:
    """Public release documentation."""
    return task in get_model_supported_tasks(model)


# ================================================================================

# ================================================================================

MODEL_CONFIG = {
    "uniicl": {
        "script": "eval_uniicl.py",
        "model_path": DEFAULT_UNIICL_FINETUNED_MODEL,
        "base_model_path": DEFAULT_UNIICL_BASE_MODEL,
        "use_mixed_weights": False,
    },
    "uniworld": {
        "script": "eval_uniworld_v1.py",
        "model_path": DEFAULT_UNIWORLD_MODEL,
        "flux_path": DEFAULT_FLUX_MODEL,
        "siglip_path": DEFAULT_SIGLIP_MODEL,
    },
    "qwen3vl": {
        "script": "eval_qwen3vl_vllm.py",
        "model_path": DEFAULT_QWEN3VL_MODEL,
        "tensor_parallel_size": 1,
    },
    "qwen25vl": {
        "script": "eval_qwen25vl_vllm.py",
        "model_path": DEFAULT_QWEN25VL_MODEL,
        "tensor_parallel_size": 1,
    },
    "internvl": {
        "script": "eval_internvl35.py",
        "model": DEFAULT_INTERNVL_MODEL,
        "tensor_parallel_size": 1,
    },
    "nexusgen": {
        "script": "eval_nexusgen.py",
        "model_path": DEFAULT_NEXUSGEN_MODEL,
        "flux_path": DEFAULT_FLUX_MODEL,
    },
    "ovisu1": {
        "script": "eval_ovisu1.py",
        "model_path": DEFAULT_OVISU1_MODEL,
        "max_model_len": 32768,
    },
}


HPSV3_CHECKPOINT = DEFAULT_HPSV3_CHECKPOINT
VLLM_API_BASE = DEFAULT_JUDGE_API_BASE
JUDGE_MODEL = DEFAULT_JUDGE_MODEL


# ================================================================================

# ================================================================================

def build_eval_command(
    model: str,
    task: str,
    k_shot: int,
    data_path: str,
    image_dir: str,
    output_dir: str,
    gpu_id: int = 0,
    benchmark_dir: str = ".",
    analogical_editing_eval_mode: str = "dinov3",
    no_capm: bool = False,
    capm_inject_layers: int = None,
    capm_ablation_mode: str = "none",
    capm_fixed_tau: float = None,
) -> List[str]:
    """Public release documentation."""

    if model not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model}")

    config = MODEL_CONFIG[model]
    script = config["script"]


    cmd = ["python", script]


    if model == "uniicl":
        cmd.extend([
            "--model-path", config["model_path"],
        ])
        if config.get("use_mixed_weights"):
            cmd.extend([
                "--base-model-path", config["base_model_path"],
                "--use-mixed-weights",
            ])

    elif model == "uniworld":
        cmd.extend([
            "--model-path", config["model_path"],
            "--flux-path", config["flux_path"],
            "--siglip-path", config["siglip_path"],
        ])

    elif model in ["qwen3vl", "qwen25vl"]:
        cmd.extend([
            "--model-path", config["model_path"],
            "--tensor-parallel-size", str(config["tensor_parallel_size"]),
        ])

    elif model == "internvl":
        cmd.extend([
            "--model", config["model"],
            "--tp", str(config["tensor_parallel_size"]),
        ])

    elif model == "nexusgen":
        cmd.extend([
            "--model-path", config["model_path"],
            "--flux-path", config["flux_path"],
        ])

    elif model == "ovisu1":
        cmd.extend([
            "--model-path", config["model_path"],
            "--max-model-len", str(config["max_model_len"]),
        ])


    cmd.extend([
        "--task", task,
        "--benchmark-dir", benchmark_dir,
        "--data-path", data_path,
        "--image-dir", image_dir,
        "--output-dir", output_dir,
        "--k-shot", str(k_shot),
    ])


    if model not in ["qwen3vl", "qwen25vl", "internvl"]:
        cmd.extend([
            "--judge-api-base", VLLM_API_BASE,
            "--judge-model", JUDGE_MODEL,
        ])


    if task in [
        "instructional_generation",
        "image_manipulation",
        "visual_refinement",
        "chain_of_editing",
        "analogical_editing",
        "fast_concept_generation",
    ]:
        cmd.extend(["--hps-checkpoint", HPSV3_CHECKPOINT])

    # NOTE: Do not forward analogical-editing eval-mode to model eval scripts.
    # Some deployments keep older eval entrypoints that do not accept this arg.
    _ = analogical_editing_eval_mode


    if no_capm and model == "uniicl":
        cmd.append("--no-capm")
    if capm_inject_layers is not None and model == "uniicl":
        cmd.extend(["--capm-inject-layers", str(capm_inject_layers)])
    if capm_ablation_mode != "none" and model == "uniicl":
        cmd.extend(["--capm-ablation-mode", capm_ablation_mode])
    if capm_fixed_tau is not None and model == "uniicl":
        cmd.extend(["--capm-fixed-tau", str(capm_fixed_tau)])

    return cmd


def run_single_evaluation(
    model: str,
    task: str,
    k_shot: int,
    gpu_id: int = 0,
    benchmark_dir: str = ".",
    output_base_dir: str = OUTPUT_BASE_DIR,
    log_base_dir: str = LOG_BASE_DIR,
    analogical_editing_eval_mode: str = "dinov3",
    analogical_editing_data_path: str = "",
    no_capm: bool = False,
    capm_inject_layers: int = None,
    capm_ablation_mode: str = "none",
    capm_fixed_tau: float = None,
) -> int:
    """Public release documentation."""


    if not is_task_supported(model, task):
        print(f"[WARN] Task '{task}' is not supported for model '{model}'")
        return 1


    try:
        if task == "analogical_editing" and analogical_editing_data_path:
            data_path = analogical_editing_data_path
        else:
            data_path = get_task_data_path(task, benchmark_dir)
        image_dir = get_task_image_dir(task)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1


    output_dir = os.path.join(output_base_dir, model, f"{k_shot}shot")
    os.makedirs(output_dir, exist_ok=True)


    log_dir = os.path.join(log_base_dir, model)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{task}_{k_shot}shot.log")


    cmd = build_eval_command(
        model, task, k_shot, data_path, image_dir, output_dir, gpu_id, benchmark_dir,
        analogical_editing_eval_mode=analogical_editing_eval_mode,
        no_capm=no_capm,
        capm_inject_layers=capm_inject_layers,
        capm_ablation_mode=capm_ablation_mode,
        capm_fixed_tau=capm_fixed_tau,
    )


    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


    print(f"[{datetime.now().strftime('%H:%M:%S')}] [GPU {gpu_id}] Starting: {task} ({k_shot}-shot)")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_file}")

    with open(log_file, 'w') as f:
        try:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False
            )

            if result.returncode == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [GPU {gpu_id}] [OK] Completed: {task} ({k_shot}-shot)")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [GPU {gpu_id}] [FAIL] Failed: {task} ({k_shot}-shot) - exit {result.returncode}")

            return result.returncode

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [GPU {gpu_id}] [ERROR] Exception: {task} ({k_shot}-shot) - {e}")
            f.write(f"\nException: {e}\n")
            return 1


def run_ablation(
    model: str,
    task: str,
    gpu_id: int = 0,
    benchmark_dir: str = ".",
    output_base_dir: str = OUTPUT_BASE_DIR,
    log_base_dir: str = LOG_BASE_DIR,
    analogical_editing_eval_mode: str = "dinov3",
    analogical_editing_data_path: str = "",
    no_capm: bool = False,
    capm_inject_layers: int = None,
    capm_ablation_mode: str = "none",
    capm_fixed_tau: float = None,
) -> int:
    """Public release documentation."""

    k_shot_range = get_task_kshot_range(task)

    print(f"=== K-shot Ablation Study (task: {task}) ===")
    print(f"K-shot range: {k_shot_range}")
    print()

    failed_count = 0

    for k_shot in k_shot_range:
        print("=" * 60)
        print(f"Running {k_shot}-shot evaluation")
        print("=" * 60)

        ret = run_single_evaluation(
            model, task, k_shot, gpu_id, benchmark_dir, output_base_dir, log_base_dir,
            analogical_editing_eval_mode=analogical_editing_eval_mode,
            analogical_editing_data_path=analogical_editing_data_path,
            no_capm=no_capm,
            capm_inject_layers=capm_inject_layers,
            capm_ablation_mode=capm_ablation_mode,
            capm_fixed_tau=capm_fixed_tau,
        )

        if ret != 0:
            failed_count += 1

        print()

    if failed_count == 0:
        print(f"[OK] Ablation study completed! All {len(k_shot_range)} evaluations succeeded.")
    else:
        print(f"[WARN] Ablation study completed with {failed_count}/{len(k_shot_range)} failures.")

    print(f"Results: {output_base_dir}/{model}/[K]shot/")

    return failed_count


def run_parallel(
    model: str,
    k_shot: int = 0,
    ablation: bool = False,
    benchmark_dir: str = ".",
    output_base_dir: str = OUTPUT_BASE_DIR,
    log_base_dir: str = LOG_BASE_DIR,
    analogical_editing_eval_mode: str = "dinov3",
    analogical_editing_data_path: str = "",
    no_capm: bool = False,
    capm_inject_layers: int = None,
    capm_ablation_mode: str = "none",
    capm_fixed_tau: float = None,
) -> int:
    """Public release documentation."""

    print("=" * 80)
    print("=== Multi-GPU Parallel Execution ===")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"K-shot: {k_shot if not ablation else 'ablation'}")
    if model == "uniicl":
        print(f"CAPM ablation mode: {capm_ablation_mode}")
        if capm_fixed_tau is not None:
            print(f"CAPM fixed tau: {capm_fixed_tau}")
    print()


    for gpu, tasks in GPU_TASKS.items():
        print(f"  GPU {gpu}: {' '.join(tasks)}")
    print("=" * 80)
    print()


    supported_tasks = get_model_supported_tasks(model)

    model_log_dir = os.path.join(log_base_dir, model)
    os.makedirs(model_log_dir, exist_ok=True)

    processes = []
    script_path = str(Path(__file__).resolve())

    for gpu, tasks in GPU_TASKS.items():

        gpu_tasks = [t for t in tasks if t in supported_tasks]

        if not gpu_tasks:
            print(f"[GPU {gpu}] All tasks unsupported for model '{model}', skip.")
            continue

        print(f"[GPU {gpu}] Tasks: {' '.join(gpu_tasks)}")

        cmd = [
            sys.executable,
            script_path,
            "--model", model,
            "--task", *gpu_tasks,
            "--gpu", str(gpu),
            "--benchmark-dir", benchmark_dir,
            "--output-dir", output_base_dir,
            "--log-dir", log_base_dir,
        ]
        if analogical_editing_data_path:
            cmd.extend(["--analogical-editing-data-path", analogical_editing_data_path])

        if no_capm:
            cmd.append("--no-capm")
        if capm_inject_layers is not None:
            cmd.extend(["--capm-inject-layers", str(capm_inject_layers)])
        if capm_ablation_mode != "none":
            cmd.extend(["--capm-ablation-mode", capm_ablation_mode])
        if capm_fixed_tau is not None:
            cmd.extend(["--capm-fixed-tau", str(capm_fixed_tau)])

        if ablation:
            cmd.append("--ablation")
            queue_log = os.path.join(model_log_dir, f"gpu{gpu}_ablation.log")
        else:
            cmd.extend(["--k-shot", str(k_shot)])
            queue_log = os.path.join(model_log_dir, f"gpu{gpu}_queue.log")

        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(gpu)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        log_handle = open(queue_log, "w")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

        processes.append(
            {
                "gpu": gpu,
                "proc": proc,
                "log_handle": log_handle,
                "queue_log": queue_log,
            }
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU {gpu} started (PID: {proc.pid})")
        print(f"  Queue log: {queue_log}")

    if not processes:
        print("\n[WARN] No runnable tasks found for this model in parallel mode.")
        return 1

    print()
    print("All GPUs launched. Waiting for completion...")
    print("=" * 80)

    failed_queues = 0
    try:
        for item in processes:
            ret = item["proc"].wait()
            item["log_handle"].close()

            if ret == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [GPU {item['gpu']}] [OK] Queue completed")
            else:
                failed_queues += 1
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] [GPU {item['gpu']}] [FAIL] Queue failed "
                    f"(exit {ret}), check {item['queue_log']}"
                )
    finally:
        for item in processes:
            if not item["log_handle"].closed:
                item["log_handle"].close()

    print("\n" + "=" * 80)
    if failed_queues == 0:
        print("[OK] All GPU queues completed!")
    else:
        print(f"[WARN] Parallel execution completed with {failed_queues}/{len(processes)} failed queues.")
    print(f"Results: {os.path.join(output_base_dir, model)}/")
    print(f"Logs: {model_log_dir}/")
    print("=" * 80)

    return failed_queues


# ================================================================================

# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="UniICL-Bench Evaluation Script (Python Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-task evaluation
  %(prog)s --model ovisu1 --task image_manipulation --k-shot 8

  # Ablation mode (all k-shot values with one model load per run)
  %(prog)s --model ovisu1 --task image_manipulation --ablation

  # Multi-task ablation
  %(prog)s --model ovisu1 --task image_manipulation instructional_generation style_aware_caption --ablation

  # Multi-task evaluation with a fixed k-shot
  %(prog)s --model ovisu1 --task image_manipulation instructional_generation --k-shot 8

  # Evaluate all tasks in parallel
  %(prog)s --model ovisu1 --parallel
        """
    )

    parser.add_argument(
        "--model", "--model-type",
        type=str,
        required=True,
        choices=list(MODEL_CONFIG.keys()),
        help="Model to evaluate"
    )

    parser.add_argument(
        "--task",
        type=str,
        nargs='+',
        help="Task name(s) using paper-aligned snake_case names (for example: --task image_manipulation instructional_generation)"
    )

    parser.add_argument(
        "--k-shot",
        type=int,
        default=0,
        help="Number of shots (default: 0)"
    )

    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run K-shot ablation study (all K values)"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tasks in parallel on multiple GPUs"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=int(os.environ.get("HIP_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",")[0]),
        help="GPU ID to use (default: from HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES or 0)"
    )

    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=".",
        help="UniICL-Bench directory (default: .)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_BASE_DIR,
        help=f"Output directory (default: {OUTPUT_BASE_DIR})"
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=LOG_BASE_DIR,
        help=f"Log directory (default: {LOG_BASE_DIR})"
    )
    parser.add_argument(
        "--analogical-editing-eval-mode",
        dest="analogical_editing_eval_mode",
        type=str,
        default="dinov3",
        choices=["dinov3", "mllm"],
        help="Evaluation mode for analogical_editing when model is UniICL (default: dinov3)"
    )
    parser.add_argument(
        "--visualcloze-g-eval-mode",
        dest="analogical_editing_eval_mode",
        type=str,
        choices=["dinov3", "mllm"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--analogical-editing-data-path",
        dest="analogical_editing_data_path",
        type=str,
        default="",
        help="Optional override data path for analogical_editing (supports .json/.jsonl)"
    )
    parser.add_argument(
        "--visualcloze-g-data-path",
        dest="analogical_editing_data_path",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-capm",
        action="store_true",
        help="Disable CAPM module (only affects UniICL)"
    )
    parser.add_argument(
        "--capm-inject-layers",
        type=int,
        default=None,
        help="Override CAPM injection layers (only affects UniICL). "
             "Typical values: 28 (full), 14 (top-half), 7 (top-quarter), 0 (disabled)"
    )
    parser.add_argument(
        "--capm-ablation-mode",
        type=str,
        default="none",
        choices=CAPM_ABLATION_CHOICES,
        help="Inference-time CAPM component ablation mode (only affects UniICL)"
    )
    parser.add_argument(
        "--capm-fixed-tau",
        type=float,
        default=None,
        help="Optional fixed routing temperature tau for UniICL CAPM. Overrides adaptive tau."
    )

    args = parser.parse_args()


    if args.parallel and args.task:
        print("[ERROR] --parallel and --task cannot be used together")
        return 1

    if not args.task and not args.parallel:
        print("[ERROR] Either --task or --parallel is required")
        return 1

    if args.task:
        try:
            args.task = [normalize_task_name(task) for task in args.task]
        except ValueError as e:
            parser.error(str(e))


    print("=" * 80)
    print("UniICL-Bench Evaluation (Python Version)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    if args.model == "uniicl":
        print(f"CAPM ablation mode: {args.capm_ablation_mode}")
        if args.capm_fixed_tau is not None:
            print(f"CAPM fixed tau: {args.capm_fixed_tau}")
    print()

    if args.parallel:

        failed = run_parallel(
            args.model,
            args.k_shot,
            args.ablation,
            args.benchmark_dir,
            args.output_dir,
            args.log_dir,
            args.analogical_editing_eval_mode,
            args.analogical_editing_data_path,
            no_capm=args.no_capm,
            capm_inject_layers=args.capm_inject_layers,
            capm_ablation_mode=args.capm_ablation_mode,
            capm_fixed_tau=args.capm_fixed_tau,
        )
        return 1 if failed else 0
    else:

        tasks = args.task
        print(f"Tasks to run: {', '.join(tasks)}")
        print()

        failed_tasks = []

        for task in tasks:
            print(f"\n{'='*80}")
            print(f"Running task: {task}")
            print(f"{'='*80}\n")

            if args.ablation:

                result = run_ablation(
                    args.model,
                    task,
                    args.gpu,
                    args.benchmark_dir,
                    args.output_dir,
                    args.log_dir,
                    args.analogical_editing_eval_mode,
                    args.analogical_editing_data_path,
                    no_capm=args.no_capm,
                    capm_inject_layers=args.capm_inject_layers,
                    capm_ablation_mode=args.capm_ablation_mode,
                    capm_fixed_tau=args.capm_fixed_tau,
                )
            else:

                result = run_single_evaluation(
                    args.model,
                    task,
                    args.k_shot,
                    args.gpu,
                    args.benchmark_dir,
                    args.output_dir,
                    args.log_dir,
                    args.analogical_editing_eval_mode,
                    args.analogical_editing_data_path,
                    no_capm=args.no_capm,
                    capm_inject_layers=args.capm_inject_layers,
                    capm_ablation_mode=args.capm_ablation_mode,
                    capm_fixed_tau=args.capm_fixed_tau,
                )

            if result != 0:
                failed_tasks.append(task)


        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")
        print(f"Total tasks: {len(tasks)}")
        print(f"Successful: {len(tasks) - len(failed_tasks)}")
        print(f"Failed: {len(failed_tasks)}")
        if failed_tasks:
            print(f"Failed tasks: {', '.join(failed_tasks)}")
        print(f"{'='*80}\n")

        return 1 if failed_tasks else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
