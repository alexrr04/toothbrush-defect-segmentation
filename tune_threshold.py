import csv
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src.unet import UNet


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
    parser = ArgumentParser(description="Tune inference threshold on a chosen split.")
    parser.add_argument(
        "--csv",
        default=os.path.join("data", "toothbrush_dataset", "validation.csv"),
        help="CSV split used for threshold tuning (default: validation.csv)",
    )
    parser.add_argument(
        "--weights",
        default=os.path.join("trained_models", "best_unet_model.pth"),
        help="Path to model weights checkpoint to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where threshold tuning artifacts are stored. Defaults to checkpoint directory.",
    )
    args = parser.parse_args()

    split_csv = args.csv
    if not os.path.exists(split_csv):
        print(f"Error: {split_csv} not found. Run prepare_data.py first.")
        return

    weights_path = args.weights
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found: {weights_path}")
        return

    output_dir = args.output_dir or os.path.dirname(weights_path) or "."
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(in_channels=3, out_channels=1).to(device)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.eval()

    print(f"Using device: {device}")
    print(f"Loaded model weights: {weights_path}")

    csv_name = os.path.basename(split_csv).lower()
    if csv_name == "testing.csv":
        print("Warning: tuning on testing.csv can overfit local evaluation.")

    df = pd.read_csv(split_csv)
    print(f"Loaded {len(df)} images from {split_csv}")

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

        h, w = img.shape[:2]
        img_f = img.astype(np.float32)
        max_val = float(np.max(img_f))
        if max_val > 1.5:
            img_f = img_f / 255.0

        img_tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(device)
        img_tensor = F.interpolate(
            img_tensor, size=(256, 256), mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            logits = net(img_tensor)
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
    best_iou = float(np.mean(results[best_threshold]["iou"]))
    best_f1 = float(np.mean(results[best_threshold]["f1"]))
    print(f"   Mean IoU: {best_iou:.4f}")
    print(f"   Mean F1:  {best_f1:.4f}")

    rows = []
    for threshold in sorted(thresholds):
        rows.append(
            {
                "threshold": float(threshold),
                "mean_iou": float(np.mean(results[threshold]["iou"])),
                "mean_f1": float(np.mean(results[threshold]["f1"])),
            }
        )

    csv_path = output_dir_path / "threshold_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["threshold", "mean_iou", "mean_f1"])
        writer.writeheader()
        writer.writerows(rows)

    best_path = output_dir_path / "best_threshold.json"
    with open(best_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "weights_path": str(weights_path),
                "csv_split": str(split_csv),
                "best_threshold": float(best_threshold),
                "mean_iou": best_iou,
                "mean_f1": best_f1,
            },
            file,
            indent=2,
            sort_keys=True,
        )

    print(f"\nSaved threshold table to: {csv_path}")
    print(f"Saved best threshold json to: {best_path}")
    print(f"\nUpdate threshold in model.py to {best_threshold:.2f}")


if __name__ == "__main__":
    main()
