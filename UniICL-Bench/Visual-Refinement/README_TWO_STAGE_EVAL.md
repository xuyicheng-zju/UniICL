# Visual Refinement Two-Stage Evaluation

## Problem
The `visual_refinement` task requires Q-Align scoring, but the Q-Align runtime can conflict with the main inference environment.

## Recommended workflow
Split evaluation into two stages:
1. **Generation stage**: run inference only and skip Q-Align loading.
2. **Scoring stage**: score the generated images in a separate environment.

## Stage 1: generate images
Run evaluation with `--skip-visual-refinement-scoring`:

```bash
python run_eval.py --model uniicl --task visual_refinement --skip-visual-refinement-scoring
```

Or call the Python entrypoint directly:

```bash
python eval_uniicl.py \
    --task visual_refinement \
    --benchmark-dir ./UniICL-Bench \
    --output-dir ./eval_results/uniicl \
    --k-shot 0 \
    --skip-visual-refinement-scoring
```

Outputs:
- Generated images: `./eval_results/uniicl/0shot/visual_refinement_generated/`
- Manifest: `./eval_results/uniicl/0shot/visual_refinement_results.json`

## Stage 2: score the generated images

```bash
python score_perfection_images.py \
    --generated-dir ./eval_results/uniicl/0shot/visual_refinement_generated \
    --data-path ./UniICL-Bench/Visual-Refinement/visual_refinement_benchmark.jsonl \
    --output-path ./eval_results/uniicl/0shot/visual_refinement_scores.json \
    --device cuda:0
```

Outputs:
- Score file: `./eval_results/uniicl/0shot/visual_refinement_scores.json`
- Per-sample fields:
  - `qalign_score_input` (`score_l`)
  - `qalign_score_output`
  - `qalign_score_gt` (`score_h`)
  - `efficiency = (s_out - s_in) / (s_gt - s_in) * 100%`

## One-stage alternative
If your environment can load both the inference stack and Q-Align together, you can run the full path in one step:

```bash
python run_eval.py --model uniicl --task visual_refinement
```

## Arguments

### `eval_uniicl.py` / `run_eval.py`
- `--skip-visual-refinement-scoring`: generate outputs only and skip Q-Align scoring

### `score_perfection_images.py`
- `--generated-dir`: directory containing generated images
- `--data-path`: benchmark annotation file with `score_l` / `score_h`
- `--output-path`: output JSON path
- `--device`: Q-Align device, default `cuda:0`

## Output structure

```text
eval_results/uniicl/0shot/
├── visual_refinement_generated/
│   ├── perfected_sample_001.png
│   ├── perfected_sample_002.png
│   └── ...
├── visual_refinement_results.json
└── visual_refinement_scores.json
```

## Why use the two-stage path
- Environment isolation between inference and scoring
- Reusable generations for repeated scoring
- Flexible device placement for Q-Align
- Easier debugging of generation vs. scoring failures
