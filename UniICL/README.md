# UniICL

This directory contains the public training and inference code for UniICL.

Included components:

- `data/`: dataset builders, configs, and public `dataset_info.py`
- `modeling/`: model implementation used by training and inference
- `models/capm.py`: CAPM module
- `train/`: distributed training entrypoint and utilities
- `inferencer.py`: interleaved inference core
- `scripts/train_uniicl.sh`: unified ICL training without CAPM
- `scripts/train_uniicl_capm.sh`: unified ICL training with CAPM
- `scripts/run_uniicl_inference.py`: minimal single-sample inference script
- `scripts/prepare_uniicl_checkpoint.sh`: copy tokenizer/config files into a finetuned checkpoint dir

## Expected layout

The open-source release assumes this root structure:

```text
UniICL-OpenSource/
  UniICL-Bench/
  UniICL-760K/
  UniICL/
```

Before conversion, training, or evaluation, edit `../local_paths.py`:

```python
UNIICL_BASE_MODEL = "/path/to/base-model"
UNIICL_FINETUNED_MODEL = "/path/to/finetuned-checkpoint"
UNIICL_TARGET_CHECKPOINT = "/path/to/finetuned-checkpoint"
```

## Training

First convert the UniICL-760K public train annotations:

```bash
cd ../UniICL-760K
bash convert_unified.sh
```

Then launch training:

```bash
cd ..
python check_setup.py
cd UniICL
bash scripts/train_uniicl.sh
```

For CAPM training:

```bash
bash scripts/train_uniicl_capm.sh
```

## Inference

Understanding example:

```bash
python scripts/run_uniicl_inference.py \
  --model-path /path/to/checkpoint \
  --image /path/to/example.jpg \
  --prompt "Answer the question about this image." \
  --understanding-output
```

Generation example:

```bash
python scripts/run_uniicl_inference.py \
  --model-path /path/to/checkpoint \
  --prompt "Generate an image of a lighthouse at dusk." \
  --output-image generated.png
```
