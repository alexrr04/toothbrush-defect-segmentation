# avs-miniproject-1

Advanced Vision Systems Mini-Project: Toothbrush Defect Segmentation

## Installation

### Option 1: Basic Installation (CodaBench Compatible)

For basic inference or CodaBench deployment, install minimal dependencies:

```bash
pip install -r requirements.txt
```

### Option 2: Full Development Setup (Recommended for Local Training)

For full training pipeline and development:

```bash
pip install -r requirements-dev.txt
```

## Workflow

### 1. Fetch Dataset

Download the toothbrush dataset from Google Drive:

```bash
python fetch_data.py
```

This will:

- Download the zipped dataset
- Extract it to `data/toothbrush_dataset/`
- Clean up temporary files

### 2. Prepare Data

Split the dataset into training (80%) and testing (20%) sets:

```bash
python prepare_data.py
```

This will:

- Create `data/toothbrush_dataset/training.csv` with image and mask paths
- Create `data/toothbrush_dataset/testing.csv` for evaluation
- Stratify split by defect status for balanced datasets

### 3. Train Model

Train the U-Net model for segmentation:

```bash
python -m src.train
```

The training pipeline will:

- Load image data and corresponding masks
- Train U-Net on defective toothbrush segmentation
- Save the best model to `trained_models/best_unet_model.pth` based on test loss
- Display per-epoch training and test metrics

**Requirements:** Requires full dev setup (`requirements-dev.txt`)

### 4. Test and Verify

Run local predictions on sample images:

```bash
python local_check_predict.py
```

This will:

- Load the trained model
- Make predictions on 5 sample images from the dataset
- Display prediction statistics (shape, unique values, white ratio)
- Verify model is working correctly

## Project Structure

```bash
├── src/
│   ├── train.py              # Training loop
│   ├── dataset.py            # Dataset class
│   ├── unet.py               # U-Net model architecture
│   └── __init__.py
├── data/                       # Dataset directory (created by fetch_data.py)
├── trained_models/             # Trained model weights
│   └── best_unet_model.pth    # Best trained model
├── fetch_data.py              # Download dataset from Google Drive
├── prepare_data.py            # Prepare and split dataset
├── local_check_predict.py      # Test predictions locally
├── model.py                   # Model loading and inference utilities
├── requirements.txt           # Minimal dependencies
└── requirements-dev.txt       # Full development dependencies
```

## Notes

- The model uses U-Net architecture for semantic segmentation
- Training saves checkpoints automatically based on test loss improvement
- GPU support is automatic (CUDA if available, CPU fallback)
- All paths are relative to project root directory
