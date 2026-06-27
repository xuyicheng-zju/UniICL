# UniICL-Bench Image Dependency Index

This file marks every benchmark task that depends on external image roots, so the
public data release can later be wired to a single `UniICL-760K/` root.

## Shared External Root

Preferred layout:

```text
UniICL-760K/
  images/AIGI-Holmes/
  images/AVA/
  images/World-Aware Planning/
  images/LAION-HR/
  images/T2I/
  images/I2I/
  images/degraded/
  images/Concept/
  images/Chain-of-Editing/
```

The public release assumes `UniICL-Bench/` and `UniICL-760K/` sit under the
same repository root, so no extra path variable is required.

## Per-Task Dependency Summary

| Task | UniICL-Bench File | Image Fields / Pattern | Expected Location |
|---|---|---|---|
| `visual_grounding` | `Visual-Grounding/visual_grounding_benchmark.jsonl` | `image_name`, `demos[].image_name` | `UniICL-760K/images/LAION-HR/<filename>` |
| `attribute_recognition` | `Attribute-Recognition/attribute_recognition_benchmark.jsonl` | `image_name`, `demos[].image_name` | `UniICL-760K/images/LAION-HR/<filename>` |
| `scene_reasoning` | `Scene-Reasoning/scene_reasoning_benchmark.jsonl` | `image_name`, `demos[].image_name` | `UniICL-760K/images/LAION-HR/<filename>` |
| `style_aware_caption` | `Style-Aware-Caption/style_aware_caption_benchmark.jsonl` | `image_name`, `demos[].image_name` | `UniICL-760K/images/LAION-HR/<filename>` |
| `analogical_inference` | `Analogical-Inference/analogical_inference_benchmark.jsonl` | `image_name`, `demos[].image_name` | `UniICL-760K/images/LAION-HR/<filename>` |
| `instructional_generation` | `Instructional-Generation/instructional_generation_benchmark.jsonl` | `image_name`, `answer`, `demos[].image_name`, `demos[].answer` | `UniICL-760K/images/T2I/...` |
| `image_manipulation` | `Image-Manipulation/image_manipulation_benchmark.jsonl` | `image_name`, `answer`, `demos[].image_name`, `demos[].answer` | `UniICL-760K/images/T2I/...` and `UniICL-760K/images/I2I/...` |
| `visual_refinement` | `Visual-Refinement/visual_refinement_benchmark.jsonl` | `image_name`, `answer`, `demos[].image_name`, `demos[].answer` | `UniICL-760K/images/degraded/...` and `UniICL-760K/images/T2I/...` |
| `analogical_editing` | `Analogical-Editing/analogical_editing_benchmark.json` | `query.input`, `query.output`, `demo[].input`, `demo[].output` | `UniICL-760K/images/T2I/...` and `UniICL-760K/images/I2I/...` |
| `aesthetic_assessment` | `Aesthetic-Assessment/aesthetic_assessment_benchmark.jsonl` | `image_name`, `demos[].image_name` | `UniICL-760K/images/AVA/<filename>` |
| `forgery_detection` | `Forgery-Detection/forgery_detection_benchmark.jsonl` | `image_name`, `demos[].image_name` | `UniICL-760K/images/AIGI-Holmes/<filename>` |
| `world_aware_planning` | `World-Aware-Planning/world_aware_planning_benchmark.json` | `images[]` | `UniICL-760K/images/World-Aware Planning/<filename>` |
| `fast_concept_mapping` | `Fast-Concept-Mapping/fast_concept_mapping_benchmark.json` | `query.image`, `demos[].image` | `UniICL-760K/images/Concept/<filename>` |
| `fast_concept_generation` | `Fast-Concept-Generation/fast_concept_generation_benchmark.json` | `query.image`, `demos[].image` | `UniICL-760K/images/Concept/<filename>` |
| `chain_of_editing` | `Chain-of-Editing/chain_of_editing_benchmark.json` | `original_image`, `edit_steps[].reference_image` | `UniICL-760K/images/Chain-of-Editing/<filename>` |

## Code Paths That Resolve Image Roots

- `UniICL-Bench/public_path_config.py`
- `UniICL-Bench/run_eval.py`
- `UniICL-Bench/verify_all_tasks.py`
- model-specific eval entrypoints under `UniICL-Bench/eval_*.py`

## Confirmed Naming Choice

- `degraded` is intentionally lowercase.
- The canonical public location is `UniICL-760K/images/degraded/...`.
- Every benchmark annotation now stores paths relative to `UniICL-760K/`, so the
  runtime image root should be the `UniICL-760K` directory itself.
