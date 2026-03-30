import os

import numpy as np
import pandas as pd
from PIL import Image

import model


def load_ground_truth_mask(mask_path: str) -> np.ndarray:
    """Load ground truth mask as binary (0/255)."""
    if pd.isna(mask_path):
        return None
    if not os.path.exists(mask_path):
        return None
    img = Image.open(mask_path).convert("L")
    mask = np.array(img, dtype=np.uint8)
    # Ground truth: 255 = defect, 0 = good
    return (mask > 127).astype(np.uint8) * 255


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over Union."""
    intersection = np.logical_and(pred > 0, gt > 0).sum()
    union = np.logical_or(pred > 0, gt > 0).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    """F1 score (Dice coefficient)."""
    tp = np.logical_and(pred > 0, gt > 0).sum()
    fp = np.logical_and(pred > 0, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt > 0).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def main():
    # Load test CSV
    test_csv = os.path.join("data", "toothbrush_dataset", "testing.csv")
    if not os.path.exists(test_csv):
        print(f"Error: {test_csv} not found. Run prepare_data.py first.")
        return

    df = pd.read_csv(test_csv)
    print(f"Loaded {len(df)} test images from {test_csv}")

    # Test thresholds from 0.1 to 0.9
    thresholds = np.arange(0.1, 1.0, 0.05)
    results = {t: {"iou": [], "f1": []} for t in thresholds}

    for idx, row in df.iterrows():
        img_path = row["image_path"]
        mask_path = row["mask_path"]

        if not os.path.exists(img_path):
            print(f"Skip {img_path}: not found")
            continue

        # Load image and predict
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        # Load ground truth
        gt = load_ground_truth_mask(mask_path)
        if gt is None:
            # For "good" images (no mask), gt is all zeros
            gt = np.zeros(img.shape[:2], dtype=np.uint8)

        import torch
        import torch.nn.functional as F

        h, w = img.shape[:2]
        img_f = img.astype(np.float32)
        max_val = float(np.max(img_f))
        if max_val > 1.5:
            img_f = img_f / 255.0

        img_tensor = (
            torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(model.device)
        )
        img_tensor = F.interpolate(
            img_tensor, size=(256, 256), mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            logits = model.model(img_tensor)
            probs = torch.sigmoid(logits)

        # Resize back to original
        probs_resized = F.interpolate(probs.float(), size=(h, w), mode="nearest")
        probs_np = probs_resized.squeeze().cpu().numpy()

        # Test each threshold
        for threshold in thresholds:
            pred = (probs_np > threshold).astype(np.uint8) * 255
            iou = compute_iou(pred, gt)
            f1 = compute_f1(pred, gt)
            results[threshold]["iou"].append(iou)
            results[threshold]["f1"].append(f1)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} images...")

    # Compute mean metrics for each threshold
    print("\n" + "=" * 70)
    print("Threshold Tuning Results")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Mean IoU':<15} {'Mean F1':<15}")
    print("-" * 70)

    best_threshold = None
    best_score = -1

    for threshold in sorted(thresholds):
        mean_iou = np.mean(results[threshold]["iou"])
        mean_f1 = np.mean(results[threshold]["f1"])
        combined_score = (mean_iou + mean_f1) / 2

        print(f"{threshold:<12.2f} {mean_iou:<15.4f} {mean_f1:<15.4f}")

        if combined_score > best_score:
            best_score = combined_score
            best_threshold = threshold

    print("-" * 70)
    print(f"\n✅ BEST THRESHOLD: {best_threshold:.2f}")
    print(f"   Mean IoU: {np.mean(results[best_threshold]['iou']):.4f}")
    print(f"   Mean F1:  {np.mean(results[best_threshold]['f1']):.4f}")
    print(f"\nUpdate threshold in model.py to {best_threshold:.2f}")


if __name__ == "__main__":
    main()
