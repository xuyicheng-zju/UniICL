# Analogical Inference

`Analogical Inference` is the public benchmark split for analogy-based
understanding tasks in UniICL-Bench.

## Public UniICL-Bench File

- `UniICL-Bench/Analogical-Inference/analogical_inference_benchmark.jsonl`

## Image Root

This task now stores root-relative image references such as
`images/LAION-HR/000601256338.jpg`, so public users should point the benchmark
to:

- `UniICL-760K`

The benchmark code resolves Analogical Inference through the shared
`LAION-HR` image category defined in `UniICL-Bench/public_path_config.py`.

## Sample Format

```json
{
  "image_name": "images/LAION-HR/000601256338.jpg",
  "intent": "bbox_specific:poster",
  "text": "poster",
  "answer": "{\"poster\": [0.0, 0.0, 1.0, 1.0]}",
  "annotation": "{\"poster\": [0.0, 0.0, 1.0, 1.0]}",
  "task_type": "bbox_specific",
  "demos": [
    {
      "image_name": "images/LAION-HR/000530915073.jpg",
      "intent": "bbox_specific:poster",
      "text": "poster",
      "annotation": "{\"poster\": [0.081, 0.055, 0.689, 0.926]}"
    }
  ]
}
```

## Evaluation

Use the standard benchmark entrypoints:

- `UniICL-Bench/run_eval.py`
- model-specific `UniICL-Bench/eval_*.py` scripts
- `UniICL-Bench/verify_all_tasks.py` for image-path validation
