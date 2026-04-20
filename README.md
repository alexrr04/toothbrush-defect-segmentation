# toothbrush-defect-segmentation

Advanced Vision Systems course project: toothbrush defect segmentation with U-Net.

## Installation

### 1. Inference-only setup

```bash
pip install -r requirements.txt
```

### 2. Full training/development setup

```bash
pip install -r requirements-dev.txt
```

## End-to-End Workflow

### 1. Fetch dataset

```bash
python fetch_data.py
```

This downloads and extracts the dataset to `data/toothbrush_dataset/`.

### 2. Prepare CSV splits

```bash
python prepare_data.py
```

This creates:

- `data/toothbrush_dataset/training.csv`
- `data/toothbrush_dataset/validation.csv`
- `data/toothbrush_dataset/testing.csv`

### 3. Train one model

```bash
python src/train.py
```

Best checkpoint is saved under the run directory created by the training config
(for example in `runs/.../best_unet_model.pth`).

## Prediction With One or More Trained Models

Inference is performed by `model.predict(image)` in `model.py`.

### Model resolution order

At import time, `model.py` resolves model paths in this order:

1. `trained_models/ensemble_weights.txt`
2. `AVS_ENSEMBLE_WEIGHTS` environment variable
3. fallback single model search for `best_unet_model.pth`

If none are found, it warns and uses randomly initialized weights.

### Use multiple models (ensemble)

Create or edit `trained_models/ensemble_weights.txt` with one checkpoint path per line:

```text
# Relative paths from repository root
trained_models/A_best.pth
trained_models/C_best.pth
```

Rules:

- Empty lines are ignored.
- Lines starting with `#` are treated as comments.
- Use relative paths for portable submission packaging.

When more than one checkpoint is provided, logits are averaged before thresholding.

### Override models from shell (optional)

Useful for quick local experiments without editing files:

```bash
export AVS_ENSEMBLE_WEIGHTS="trained_models/A_best.pth,trained_models/D_best.pth"
python local_check_predict.py
```

### Tune threshold for selected model set

```bash
python tune_threshold.py --weights-file trained_models/ensemble_weights.txt --csv data/toothbrush_dataset/validation.csv --output-dir trained_models/tune_custom
```

Artifacts produced:

- `threshold_results.csv`
- `best_threshold.json`

### Apply tuned threshold at inference

`model.py` defaults to threshold `0.65`. You can override it without code edits:

```bash
export AVS_THRESHOLD=0.60
python local_check_predict.py
```

## Local Inference Sanity Check

Run prediction on a few training images:

```bash
python local_check_predict.py
```

This prints image/mask stats and validates mask format (`uint8`, values in `{0,255}`).

## Submission Packaging

Create upload archive:

```bash
python make_submission.py
```

Packaging behavior:

- Requires `trained_models/ensemble_weights.txt`.
- Includes only checkpoint files listed there.
- Includes code and metadata needed for reproducible inference.
- Skips hidden files, data directory contents, PDFs, and `trained_models/tune_*` artifacts.

## Project Structure

```text
.
├── configs/
├── src/
│   ├── dataset.py
│   ├── train.py
│   └── unet.py
├── fetch_data.py
├── prepare_data.py
├── model.py
├── local_check_predict.py
├── tune_threshold.py
├── make_submission.py
├── requirements.txt
├── requirements-dev.txt
└── trained_models/
    ├── ensemble_weights.txt
    └── ... local checkpoints (.pth, ignored by git)
```

## Notes

- GPU is used automatically when CUDA is available.
- Keep large datasets/checkpoints outside version control.
- For reproducible experiments, keep `ensemble_weights.txt` and threshold tuning artifacts per run.
