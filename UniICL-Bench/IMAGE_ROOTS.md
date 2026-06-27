# UniICL-Bench Image Roots

The public benchmark package keeps benchmark JSON/JSONL definitions in-place, while
the actual images are expected under a shared `UniICL-760K/` root:

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

## Preferred Setup

Keep the benchmark repo and the dataset repo in the same release root:

```text
UniICL-OpenSource/
  UniICL-Bench/
  UniICL-760K/
```

With this layout, the benchmark code resolves all task image paths automatically.

## Image Resolution Rule

The public release expects every annotation path to stay relative to
`UniICL-760K/`, for example `images/LAION-HR/000123.jpg` or
`images/T2I/sample_000001.png`.

Generated-image scorers also read from `UniICL-760K/images`, so no extra path
override is required in the public release.

## Task Mapping

| Task | UniICL-Bench File | Image Root |
|---|---|---|
| `visual_grounding` | `Visual-Grounding/visual_grounding_benchmark.jsonl` | `UniICL-760K` |
| `attribute_recognition` | `Attribute-Recognition/attribute_recognition_benchmark.jsonl` | `UniICL-760K` |
| `scene_reasoning` | `Scene-Reasoning/scene_reasoning_benchmark.jsonl` | `UniICL-760K` |
| `style_aware_caption` | `Style-Aware-Caption/style_aware_caption_benchmark.jsonl` | `UniICL-760K` |
| `analogical_inference` | `Analogical-Inference/analogical_inference_benchmark.jsonl` | `UniICL-760K` |
| `instructional_generation` | `Instructional-Generation/instructional_generation_benchmark.jsonl` | `UniICL-760K` |
| `image_manipulation` | `Image-Manipulation/image_manipulation_benchmark.jsonl` | `UniICL-760K` |
| `visual_refinement` | `Visual-Refinement/visual_refinement_benchmark.jsonl` | `UniICL-760K` |
| `analogical_editing` | `Analogical-Editing/analogical_editing_benchmark.json` | `UniICL-760K` |
| `aesthetic_assessment` | `Aesthetic-Assessment/aesthetic_assessment_benchmark.jsonl` | `UniICL-760K` |
| `forgery_detection` | `Forgery-Detection/forgery_detection_benchmark.jsonl` | `UniICL-760K` |
| `world_aware_planning` | `World-Aware-Planning/world_aware_planning_benchmark.json` | `UniICL-760K` |
| `fast_concept_mapping` | `Fast-Concept-Mapping/fast_concept_mapping_benchmark.json` | `UniICL-760K` |
| `fast_concept_generation` | `Fast-Concept-Generation/fast_concept_generation_benchmark.json` | `UniICL-760K` |
| `chain_of_editing` | `Chain-of-Editing/chain_of_editing_benchmark.json` | `UniICL-760K` |

## Notes

- `instructional_generation`, `image_manipulation`, `visual_refinement`, and `analogical_editing` store relative paths such as
  `images/T2I/...`, `images/I2I/...`, and `images/degraded/...`.
- Understanding tasks now also use root-relative paths such as
  `images/LAION-HR/...`, `images/AVA/...`, and `images/AIGI-Holmes/...`.
- `degraded` is intentionally lowercase and should stay as `UniICL-760K/images/degraded`.
- `run_eval.py`, `verify_all_tasks.py`, and the model-specific eval entrypoints
  all read from `public_path_config.py`.
