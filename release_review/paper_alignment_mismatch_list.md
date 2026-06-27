# UniICL Paper Alignment Check

Source of truth:
- Paper: `UniICL.pdf`
- Open-source release root: `UniICL-OpenSource/`

## Already aligned

- The legacy module name has been fully removed from code, scripts, docs, and file names in favor of `CAPM`.
- The CAPM expansion now matches the paper: `Context-Adaptive Prototype Modulator`.
- Public release top-level names match the paper:
  - `UniICL-760K`
  - `UniICL-Bench`
- Public dataset and benchmark directories now use the paper task names.
- Public benchmark file names now use the paper task names.
- Benchmark-facing print strings have been updated to the paper task names.
- A paper-to-release taxonomy mapping is now available in `UniICL-Bench/README.md`.
- The main benchmark release matches the paper's `1,250`-episode claim.

## Remaining mismatches against the paper

### 1. `UniICL-760K` is a family name, not the exact released sample count

The semantically decontaminated release annotations sum to `747,015` samples.
The public documentation now states this explicitly, but the top-level dataset
name remains `UniICL-760K` as the release family name.

Current public documentation:
- `README.md`
- `UniICL-760K/README.md`

Status:
- Documented and intentional.

### 2. Internal model/evaluator compatibility still exposes `Bagel`

The public-facing layer is now `UniICL`, but the compatibility layer still
retains `Bagel` in the implementation for checkpoint and runtime stability.

Primary locations:
- `UniICL/modeling/uniicl/bagel.py`
- `UniICL/modeling/uniicl/__init__.py`
- `UniICL/train/pretrain_unified_navit.py`
- `UniICL-Bench/eval_bagel.py`
- `UniICL-Bench/eval_uniicl.py`
- `UniICL-Bench/eval_uniworld_v1.py`

Status:
- Intentional compatibility debt.
- Safe to keep for now because it does not affect the public release names or
  paper-aligned task presentation.

### 3. Image asset category names remain source-oriented rather than task-oriented

The paper talks about tasks, but the shared image root uses source/image-pool
names:

- `images/AIGI-Holmes/`
- `images/AVA/`
- `images/LAION-HR/`
- `images/T2I/`
- `images/I2I/`
- `images/degraded/`
- `images/Concept/`
- `images/World-Aware Planning/`
- `images/Chain-of-Editing/`

Status:
- Intentional and correct.
- This is an asset-layout choice, not a benchmark-task naming inconsistency.

## Current conclusion

After the latest cleanup, the remaining mismatches with the paper are limited to:

1. The nominal `UniICL-760K` brand versus the exact released count `747,015`.
2. Internal `Bagel` compatibility names kept for model/evaluator stability.
3. Source-oriented image-root category names that are intentionally separate
   from task names.
