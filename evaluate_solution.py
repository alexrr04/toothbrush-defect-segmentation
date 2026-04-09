import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay

import model


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _metrics_from_confusion(tp: int, fp: int, tn: int, fn: int) -> dict:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def _load_gt_mask(mask_path: str, shape: tuple[int, int]) -> np.ndarray:
    if not mask_path:
        return np.zeros(shape, dtype=np.uint8)
    if not os.path.exists(mask_path):
        return np.zeros(shape, dtype=np.uint8)

    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return (mask > 127).astype(np.uint8)


def _compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = int(np.logical_and(pred > 0, gt > 0).sum())
    union = int(np.logical_or(pred > 0, gt > 0).sum())
    if union == 0:
        return 1.0
    return inter / union


def _compute_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = int(np.logical_and(pred > 0, gt > 0).sum())
    fp = int(np.logical_and(pred > 0, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt > 0).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _save_pixel_confusion_matrix(
    px_tn: int, px_fp: int, px_fn: int, px_tp: int, output_dir: Path
) -> tuple[str, str]:
    cm_counts = np.array([[px_tn, px_fp], [px_fn, px_tp]], dtype=np.int64)
    cm_norm = cm_counts.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_norm / row_sums

    labels = ["Good pixel", "Defect pixel"]

    counts_path = output_dir / "pixel_confusion_matrix_counts.png"
    norm_path = output_dir / "pixel_confusion_matrix_normalized.png"

    fig_counts, ax_counts = plt.subplots(figsize=(6, 5))
    disp_counts = ConfusionMatrixDisplay(
        confusion_matrix=cm_counts, display_labels=labels
    )
    disp_counts.plot(ax=ax_counts, cmap="Blues", values_format="d", colorbar=False)
    ax_counts.set_title("Pixel-Level Confusion Matrix (Counts)")
    fig_counts.tight_layout()
    fig_counts.savefig(counts_path, dpi=160)
    plt.close(fig_counts)

    fig_norm, ax_norm = plt.subplots(figsize=(6, 5))
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp_norm.plot(ax=ax_norm, cmap="Blues", values_format=".3f", colorbar=False)
    ax_norm.set_title("Pixel-Level Confusion Matrix (Row-normalized)")
    fig_norm.tight_layout()
    fig_norm.savefig(norm_path, dpi=160)
    plt.close(fig_norm)

    return str(counts_path), str(norm_path)


def evaluate(csv_path: str, output_dir: str, top_k: int) -> dict:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)

    # Image-level confusion matrix counts
    img_tp = img_fp = img_tn = img_fn = 0

    # Pixel-level confusion matrix counts
    px_tp = px_fp = px_tn = px_fn = 0

    per_image = []

    for idx, row in enumerate(rows, start=1):
        image_path = row.get("image_path", "")
        mask_path = row.get("mask_path", "")
        if not os.path.exists(image_path):
            continue

        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        gt = _load_gt_mask(mask_path, image.shape[:2])

        pred_mask = model.predict(image)
        pred = (pred_mask > 0).astype(np.uint8)

        if pred.shape != gt.shape:
            raise ValueError(
                f"Prediction shape mismatch for {image_path}: "
                f"pred={pred.shape}, gt={gt.shape}"
            )

        # Pixel-level confusion
        px_tp += int(np.logical_and(pred == 1, gt == 1).sum())
        px_fp += int(np.logical_and(pred == 1, gt == 0).sum())
        px_tn += int(np.logical_and(pred == 0, gt == 0).sum())
        px_fn += int(np.logical_and(pred == 0, gt == 1).sum())

        # Image-level confusion (defect present or not)
        true_defective = int(gt.sum() > 0)
        pred_defective = int(pred.sum() > 0)
        if true_defective == 1 and pred_defective == 1:
            img_tp += 1
        elif true_defective == 0 and pred_defective == 1:
            img_fp += 1
        elif true_defective == 0 and pred_defective == 0:
            img_tn += 1
        else:
            img_fn += 1

        iou = _compute_iou(pred, gt)
        f1 = _compute_f1(pred, gt)

        fp_pixels = int(np.logical_and(pred == 1, gt == 0).sum())
        fn_pixels = int(np.logical_and(pred == 0, gt == 1).sum())

        per_image.append(
            {
                "index": idx,
                "image_path": image_path,
                "mask_path": mask_path,
                "true_defective": true_defective,
                "pred_defective": pred_defective,
                "iou": iou,
                "f1": f1,
                "gt_positive_pixels": int(gt.sum()),
                "pred_positive_pixels": int(pred.sum()),
                "fp_pixels": fp_pixels,
                "fn_pixels": fn_pixels,
            }
        )

    image_metrics = _metrics_from_confusion(img_tp, img_fp, img_tn, img_fn)
    pixel_metrics = _metrics_from_confusion(px_tp, px_fp, px_tn, px_fn)

    defective_rows = [r for r in per_image if r["true_defective"] == 1]
    good_rows = [r for r in per_image if r["true_defective"] == 0]

    mean_iou_all = float(np.mean([r["iou"] for r in per_image])) if per_image else 0.0
    mean_f1_all = float(np.mean([r["f1"] for r in per_image])) if per_image else 0.0
    mean_iou_defective = (
        float(np.mean([r["iou"] for r in defective_rows])) if defective_rows else 0.0
    )
    mean_f1_defective = (
        float(np.mean([r["f1"] for r in defective_rows])) if defective_rows else 0.0
    )

    misclassified_images = [
        r for r in per_image if r["true_defective"] != r["pred_defective"]
    ]
    worst_iou_defective = sorted(defective_rows, key=lambda r: r["iou"])[:top_k]
    worst_fp_good = sorted(good_rows, key=lambda r: r["fp_pixels"], reverse=True)[
        :top_k
    ]
    worst_fn_defective = sorted(
        defective_rows, key=lambda r: r["fn_pixels"], reverse=True
    )[:top_k]

    report = {
        "csv_path": csv_path,
        "n_images": len(per_image),
        "image_level": {
            "confusion_matrix": {
                "tp": img_tp,
                "fp": img_fp,
                "tn": img_tn,
                "fn": img_fn,
            },
            "metrics": image_metrics,
        },
        "pixel_level": {
            "confusion_matrix": {
                "tp": px_tp,
                "fp": px_fp,
                "tn": px_tn,
                "fn": px_fn,
            },
            "metrics": pixel_metrics,
        },
        "segmentation_summary": {
            "mean_iou_all": mean_iou_all,
            "mean_f1_all": mean_f1_all,
            "mean_iou_defective_only": mean_iou_defective,
            "mean_f1_defective_only": mean_f1_defective,
        },
        "failure_analysis": {
            "n_image_level_misclassified": len(misclassified_images),
            "misclassified_images": misclassified_images,
            "worst_iou_defective": worst_iou_defective,
            "worst_false_positives_on_good": worst_fp_good,
            "worst_false_negatives_on_defective": worst_fn_defective,
        },
    }

    cm_counts_path, cm_norm_path = _save_pixel_confusion_matrix(
        px_tn=px_tn,
        px_fp=px_fp,
        px_fn=px_fn,
        px_tp=px_tp,
        output_dir=output_dir_path,
    )
    report["pixel_level"]["confusion_matrix_plot_counts"] = cm_counts_path
    report["pixel_level"]["confusion_matrix_plot_normalized"] = cm_norm_path

    report_path = output_dir_path / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    per_image_path = output_dir_path / "per_image_metrics.csv"
    if per_image:
        with open(per_image_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(per_image[0].keys()))
            writer.writeheader()
            writer.writerows(per_image)

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate current solution metrics.")
    parser.add_argument(
        "--csv",
        default=os.path.join("data", "toothbrush_dataset", "testing.csv"),
        help="CSV split to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("analysis", "testing_eval"),
        help="Directory to save report artifacts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of worst-case examples to include in the report.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    report = evaluate(args.csv, args.output_dir, args.top_k)

    print("\nEvaluation complete")
    print(f"Split: {report['csv_path']}")
    print(f"Images evaluated: {report['n_images']}")

    img_cm = report["image_level"]["confusion_matrix"]
    img_m = report["image_level"]["metrics"]
    print("\nImage-level confusion matrix")
    print(f"TP={img_cm['tp']} FP={img_cm['fp']} TN={img_cm['tn']} FN={img_cm['fn']}")
    print(
        "Image-level metrics: "
        f"acc={img_m['accuracy']:.4f} "
        f"prec={img_m['precision']:.4f} "
        f"rec={img_m['recall']:.4f} "
        f"f1={img_m['f1']:.4f} "
        f"iou={img_m['iou']:.4f}"
    )

    px_cm = report["pixel_level"]["confusion_matrix"]
    px_m = report["pixel_level"]["metrics"]
    print("\nPixel-level confusion matrix")
    print(f"TP={px_cm['tp']} FP={px_cm['fp']} TN={px_cm['tn']} FN={px_cm['fn']}")
    print(
        "Pixel-level metrics: "
        f"acc={px_m['accuracy']:.4f} "
        f"prec={px_m['precision']:.4f} "
        f"rec={px_m['recall']:.4f} "
        f"f1={px_m['f1']:.4f} "
        f"iou={px_m['iou']:.4f}"
    )

    seg = report["segmentation_summary"]
    print("\nSegmentation summary")
    print(
        f"mean_iou_all={seg['mean_iou_all']:.4f} mean_f1_all={seg['mean_f1_all']:.4f}"
    )
    print(
        f"mean_iou_defective_only={seg['mean_iou_defective_only']:.4f} "
        f"mean_f1_defective_only={seg['mean_f1_defective_only']:.4f}"
    )

    print(
        "\nFailure summary: "
        f"misclassified_images={report['failure_analysis']['n_image_level_misclassified']}"
    )
    print(
        f"Report written to: {os.path.join(args.output_dir, 'evaluation_report.json')}"
    )
    print(
        f"Per-image table written to: {os.path.join(args.output_dir, 'per_image_metrics.csv')}"
    )


if __name__ == "__main__":
    main()
