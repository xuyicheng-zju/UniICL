# UniICL-Bench

`UniICL-Bench` is the public benchmark release for UniICL. The main release
contains `1,250` benchmark episodes spanning both understanding and generation.

## Paper-Aligned Capability Taxonomy

The benchmark follows the capability hierarchy used in the paper.

### Perception

- `Visual Grounding`
- `Attribute Recognition`
- `Image Manipulation`

### Imitation

- `Style-Aware Caption`
- `Scene Reasoning`
- `Instructional Generation`

### Conception

- `Fast Concept Mapping`
- `Fast Concept Generation`

### Deduction

- `World-Aware Planning`
- `Chain-of-Editing`

### Analogy

- `Analogical Inference`
- `Analogical Editing`

### Discernment

- `Aesthetic Assessment`
- `Forgery Detection`
- `Visual Refinement`

## Release Layout and Runtime Keys

The public directory names and benchmark file names use the paper terminology.
The public Python evaluation entrypoints now also use paper-aligned snake_case
task keys. Legacy short aliases remain accepted for backward compatibility but
are no longer part of the public interface. The mapping is:

| Paper Task Name | Benchmark Directory | Benchmark File | Runtime Task Key |
|---|---|---|---|
| `Visual Grounding` | `Visual-Grounding` | `visual_grounding_benchmark.jsonl` | `visual_grounding` |
| `Attribute Recognition` | `Attribute-Recognition` | `attribute_recognition_benchmark.jsonl` | `attribute_recognition` |
| `Scene Reasoning` | `Scene-Reasoning` | `scene_reasoning_benchmark.jsonl` | `scene_reasoning` |
| `Style-Aware Caption` | `Style-Aware-Caption` | `style_aware_caption_benchmark.jsonl` | `style_aware_caption` |
| `Instructional Generation` | `Instructional-Generation` | `instructional_generation_benchmark.jsonl` | `instructional_generation` |
| `Image Manipulation` | `Image-Manipulation` | `image_manipulation_benchmark.jsonl` | `image_manipulation` |
| `Aesthetic Assessment` | `Aesthetic-Assessment` | `aesthetic_assessment_benchmark.jsonl` | `aesthetic_assessment` |
| `Forgery Detection` | `Forgery-Detection` | `forgery_detection_benchmark.jsonl` | `forgery_detection` |
| `Visual Refinement` | `Visual-Refinement` | `visual_refinement_benchmark.jsonl` | `visual_refinement` |
| `Fast Concept Mapping` | `Fast-Concept-Mapping` | `fast_concept_mapping_benchmark.json` | `fast_concept_mapping` |
| `Fast Concept Generation` | `Fast-Concept-Generation` | `fast_concept_generation_benchmark.json` | `fast_concept_generation` |
| `World-Aware Planning` | `World-Aware-Planning` | `world_aware_planning_benchmark.json` | `world_aware_planning` |
| `Chain-of-Editing` | `Chain-of-Editing` | `chain_of_editing_benchmark.json` | `chain_of_editing` |
| `Analogical Editing` | `Analogical-Editing` | `analogical_editing_benchmark.json` | `analogical_editing` |
| `Analogical Inference` | `Analogical-Inference` | `analogical_inference_benchmark.jsonl` | `analogical_inference` |

## Evaluation

Before running evaluation, edit `../local_paths.py` and fill in at least:

```python
UNIICL_FINETUNED_MODEL = "/path/to/finetuned-checkpoint"
JUDGE_MODEL = "your-judge-model-name"
JUDGE_API_BASE = "http://127.0.0.1:8000/v1"
```

Use the shared entrypoint:

```bash
cd UniICL-Bench
python run_eval.py --model uniicl --task visual_grounding --k-shot 2
```

Image paths are resolved relative to `UniICL-760K/`; see:

- `IMAGE_ROOTS.md`
- `IMAGE_DEPENDENCIES.md`
