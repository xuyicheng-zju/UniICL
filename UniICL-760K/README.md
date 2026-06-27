# UniICL-760K

## Overview

> We introduce UniICL-760K, the first large-scale dataset specifically designed for unified multimodal In-context learning across visual understanding and generation. It contains 766,868 carefully constructed ICL episodes, each paired with a curated 8-shot demonstration context. Rather than fragmenting tasks by isolated application goals, UniICL-760K organizes understanding and generation within a six-level capability-oriented taxonomy, instantiating 15 corresponding subtasks to measure ICL capabilities across all dimensions. To scale this taxonomy-guided suite, we build an automated data curation pipeline combining dense annotation, generative augmentation, task-aligned demonstration retrieval, and strict quality control. Due to the high cost of constructing expert-level editing trajectories, the Chain-of-Editing subtask is excluded from the training corpus, retained solely in our benchmark to evaluate generative generalization. Overall, UniICL-760K serves as a scalable training resource for unified multimodal ICL, while the independently curated UniICL-Bench enables systematic evaluation.

This repository directory contains the public training release of `UniICL-760K`. The dataset is organized around the same six capability levels used in the paper:

- `Perception`
- `Imitation`
- `Conception`
- `Deduction`
- `Analogy`
- `Discernment`

Unlike a conventional supervised corpus of isolated `(image, label)` pairs, `UniICL-760K` is built as an episode-style training resource. Each sample contains a query together with in-context demonstrations, so the model must learn how to infer the task structure from examples rather than only fit the final query target.

## Release Status

The paper-level dataset family name remains `UniICL-760K`. This public release corresponds to the benchmark-aware decontaminated training split used for the released benchmark protocol.

- Full paper-scale dataset before benchmark-aware filtering: `766,868` episodes
- Public benchmark-aware release in this directory: `747,015` train samples

The released annotations therefore reflect the open-source training split after exact-overlap removal and semantic decontamination against `UniICL-Bench`. The underlying public image assets and raw task annotations can still be reused by the community to compose alternative episodes or new train/eval splits.

## Coverage

The release contains `14` training tasks:

- `9` understanding-style tasks
- `5` generation-style tasks

These tasks span the full capability spectrum targeted by UniICL:

| Level | Task | Directory | Released samples |
| --- | --- | --- | ---: |
| Perception | Visual Grounding | `Visual-Grounding/` | 66,347 |
| Perception | Attribute Recognition | `Attribute-Recognition/` | 64,338 |
| Imitation | Style-Aware Caption | `Style-Aware-Caption/` | 67,225 |
| Imitation | Scene Reasoning | `Scene-Reasoning/` | 66,074 |
| Imitation | Instructional Generation | `Instructional-Generation/` | 60,990 |
| Perception | Image Manipulation | `Image-Manipulation/` | 39,201 |
| Conception | Fast Concept Mapping | `Fast-Concept-Mapping/` | 50,000 |
| Conception | Fast Concept Generation | `Fast-Concept-Generation/` | 50,000 |
| Deduction | World-Aware Planning | `World-Aware-Planning/` | 63,964 |
| Analogy | Analogical Inference | `Analogical-Inference/` | 51,028 |
| Analogy | Analogical Editing | `Analogical-Editing/` | 18,710 |
| Discernment | Aesthetic Assessment | `Aesthetic-Assessment/` | 80,481 |
| Discernment | Forgery Detection | `Forgery-Detection/` | 40,661 |
| Discernment | Visual Refinement | `Visual-Refinement/` | 27,996 |

Release totals:

- Understanding-style release annotations: `550,118`
- Generation-style release annotations: `196,897`
- Total released annotations: `747,015`

`Chain-of-Editing` is benchmark-only and is therefore not part of this training release.

## GitHub and Hugging Face Split

The GitHub source repository is intended to track this directory's README,
conversion scripts, and packaging utilities. Large training annotations
(`*_train_icl.json/jsonl`), converted training artifacts, images, and image
archive shards should be uploaded to the Hugging Face dataset repository.

After cloning the source repository, download or sync the dataset files from
Hugging Face back into this same `UniICL-760K/` directory before running
`convert_unified.sh` or training.

## What Is Included

The full Hugging Face dataset release contains:

- train annotations for all released UniICL training tasks
- conversion scripts that transform the released annotations into the formats consumed by the UniICL training code
- no benchmark files
- no model outputs or caches
- no validation or test split

Images are not bundled inside the annotation files themselves. Instead, all image references are stored as paths relative to the `UniICL-760K/` root.

## Public Image Layout

Place images under the following shared layout:

```text
UniICL-760K/
  images/
    AIGI-Holmes/
    AVA/
    World-Aware Planning/
    LAION-HR/
    T2I/
    I2I/
    degraded/
    Concept/
    Chain-of-Editing/
```

All annotations in this release use root-relative image paths such as:

- `images/LAION-HR/000123456789.jpg`
- `images/T2I/sample_000123.png`
- `images/I2I/sample_000123.png`
- `images/degraded/sample_000123.png`
- `images/Concept/Items/Lip_Speaker/new_0021.png`
- `images/World-Aware Planning/image_029489.png`

Two details are intentional:

- `Concept` images keep their internal semantic subdirectories, because filenames are not globally unique.
- editing and refinement tasks preserve source/target pool distinctions such as `images/T2I/...`, `images/I2I/...`, and `images/degraded/...`.

## Hugging Face Release Packaging

For local training and evaluation, the expected runtime layout is still the unpacked layout shown above. However, for public Hub release we recommend shipping large image pools as archive shards instead of uploading hundreds of thousands of raw image files directly.

Recommended Hugging Face upload contents:

- `*/*_train_icl.json` and `*/*_train_icl.jsonl`
- `image_archives/**` after running `package_images_for_hf.py`
- this `README.md` and optional manifest files

Do not upload generated local conversion outputs by default:

- `*/*_uniicl.jsonl`
- `*/parquet_*`

Those files are deterministic training artifacts produced by `convert_unified.sh` after users download the release.

Recommended Hub layout:

```text
UniICL-760K/
  Aesthetic-Assessment/
  Analogical-Editing/
  ...
  image_archives/
    LAION-HR/
      laion-hr-00001.tar
      laion-hr-00002.tar
      ...
    T2I/
      t2i-00001.tar
      t2i-00002.tar
      ...
    I2I/
      i2i-00001.tar
      ...
    degraded/
      degraded-00001.tar
      ...
```

The helper script `package_images_for_hf.py` creates tar shards whose internal archive paths still follow the runtime convention `images/<target>/...`. After download, users can unpack them at the dataset root and recover the expected layout directly.

Example:

```bash
python3 package_images_for_hf.py \
  --release-root . \
  --source-root /path/to/raw_image_roots \
  --max-size-gb 9.5
```

This writes archive shards to `image_archives/` and also produces `image_archives/image_archives_manifest.json`.

## Annotation Files

Understanding-style source annotations:

- `Aesthetic-Assessment/aesthetic_assessment_train_icl.jsonl`
- `Analogical-Inference/analogical_inference_train_icl.jsonl`
- `Attribute-Recognition/attribute_recognition_train_icl.jsonl`
- `Fast-Concept-Mapping/fast_concept_mapping_train_icl.jsonl`
- `Forgery-Detection/forgery_detection_train_icl.jsonl`
- `Scene-Reasoning/scene_reasoning_train_icl.jsonl`
- `Style-Aware-Caption/style_aware_caption_train_icl.jsonl`
- `Visual-Grounding/visual_grounding_train_icl.jsonl`
- `World-Aware-Planning/world_aware_planning_train_icl.json`

Generation-style source annotations:

- `Analogical-Editing/analogical_editing_train_icl.json`
- `Fast-Concept-Generation/fast_concept_generation_train_icl.jsonl`
- `Image-Manipulation/image_manipulation_train_icl.jsonl`
- `Instructional-Generation/instructional_generation_train_icl.jsonl`
- `Visual-Refinement/visual_refinement_train_icl.jsonl`

## Training Conversion

The raw released annotations are converted into the formats expected by the UniICL training pipeline:

- understanding-style tasks become `*_uniicl.jsonl`
- generation-style tasks become `parquet_*` directories

Convert the full release:

```bash
bash convert_unified.sh
```

Convert only selected tasks:

```bash
bash convert_unified.sh --tasks visual_grounding,scene_reasoning,instructional_generation
```

The conversion script uses the public task names and writes the outputs next to the source annotations.

## Format Notes

The conversion pipeline keeps the released episode semantics intact:

- understanding-style tasks are converted into text-plus-image conversational samples for the UniICL VLM training loader
- generation-style tasks are converted into parquet shards that store image bytes together with ordered instruction lists
- `World-Aware Planning` is converted from its structured trajectory JSON into the same unified conversational format expected by the understanding loader
- `Analogical Editing` preserves multi-input edit structure through `num_inputs`

## Relationship to UniICL-Bench

This release is aligned with `UniICL-Bench` at the level of:

- task names
- path conventions
- image pool organization
- benchmark-aware decontamination policy

The released training set is therefore suitable for reproducing the public UniICL training setup while maintaining the intended isolation from the released benchmark.

## Practical Notes

- If you only want to run `UniICL-Bench`, you do not need to run `convert_unified.sh`.
- If you want to train UniICL, conversion is required before launching training.
- After training, evaluation checkpoints may still need tokenizer and autoencoder assets copied from the base model; use `UniICL/scripts/prepare_uniicl_checkpoint.sh` for that step.
