import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

# Ensure the submission root (directory of this file) is on sys.path.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from src.unet import UNet  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_inference_threshold() -> float:
    """Resolve binary threshold from env, defaulting to 0.65."""
    raw = os.environ.get("AVS_THRESHOLD", "").strip()
    if not raw:
        return 0.65

    try:
        threshold = float(raw)
    except ValueError:
        print(f"Warning: invalid AVS_THRESHOLD={raw!r}. Using default 0.65.")
        return 0.65

    if threshold <= 0.0 or threshold >= 1.0:
        print(f"Warning: AVS_THRESHOLD={threshold} outside (0, 1). Using default 0.65.")
        return 0.65

    return threshold


def _resolve_single_weights_path() -> str | None:
    """Try to locate one trained checkpoint regardless of current working directory."""
    candidates = [
        os.path.join(_BASE_DIR, "trained_models", "best_unet_model.pth"),
        os.path.join(os.getcwd(), "trained_models", "best_unet_model.pth"),
        os.path.join(_BASE_DIR, "best_unet_model.pth"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    for root, _dirs, files in os.walk(_BASE_DIR):
        if "best_unet_model.pth" in files:
            return os.path.join(root, "best_unet_model.pth")

    return None


def _resolve_weights_paths() -> list[str]:
    """Resolve one or more checkpoints for optional ensemble averaging."""
    paths = []

    # Optional text file with one checkpoint path per line.
    ensemble_file = os.path.join(_BASE_DIR, "trained_models", "ensemble_weights.txt")
    if os.path.exists(ensemble_file):
        with open(ensemble_file, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                candidate = line
                if not os.path.isabs(candidate):
                    candidate = os.path.join(_BASE_DIR, candidate)
                if os.path.exists(candidate):
                    paths.append(candidate)

    # Optional env override for local testing.
    env_paths = os.environ.get("AVS_ENSEMBLE_WEIGHTS", "").strip()
    if env_paths:
        for entry in env_paths.split(","):
            candidate = entry.strip()
            if not candidate:
                continue
            if not os.path.isabs(candidate):
                candidate = os.path.join(_BASE_DIR, candidate)
            if os.path.exists(candidate):
                paths.append(candidate)

    if paths:
        # Keep insertion order and remove duplicates.
        return list(dict.fromkeys(paths))

    single_path = _resolve_single_weights_path()
    if single_path is not None:
        return [single_path]
    return []


def _load_models(weights_paths: list[str]) -> list[UNet]:
    loaded_models = []
    for path in weights_paths:
        net = UNet(in_channels=3, out_channels=1)
        net.load_state_dict(torch.load(path, map_location=device))
        net.to(device)
        net.eval()
        loaded_models.append(net)
    return loaded_models


_weights_paths = _resolve_weights_paths()
_inference_threshold = _resolve_inference_threshold()
if _weights_paths:
    models = _load_models(_weights_paths)
    print(
        f"Loaded {len(models)} model(s) for inference ensemble "
        f"(threshold={_inference_threshold:.2f})."
    )
else:
    print(
        "Warning: Could not find trained checkpoints. "
        f"Predictions will use random weights (threshold={_inference_threshold:.2f})."
    )
    fallback_model = UNet(in_channels=3, out_channels=1)
    fallback_model.to(device)
    fallback_model.eval()
    models = [fallback_model]


def _apply_postprocessing(binary_mask: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations to clean up predictions:
    1. Median filter
    2. Erosion
    3. Dilation
    """
    filtered = ndimage.median_filter(binary_mask, size=3)

    eroded = ndimage.binary_erosion(
        filtered > 0, structure=np.ones((3, 3), dtype=np.uint8)
    ).astype(np.uint8)

    dilated = ndimage.binary_dilation(
        eroded, structure=np.ones((3, 3), dtype=np.uint8)
    ).astype(np.uint8)

    return dilated * 255


def predict(image):
    """
    Predict the segmentation mask for a given image.

    Args:
        image: numpy array, uint8. Shape (H, W, 3) RGB or (H, W) grayscale.

    Returns:
        Binary mask as numpy array of shape (H, W), uint8 with values 0 or 255.

    Notes:
        Set AVS_THRESHOLD to override the default threshold (0.65), for example:
        AVS_THRESHOLD=0.60
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    original_h, original_w = image.shape[:2]

    # Accept grayscale (H, W) or color (H, W, C). If RGBA, drop alpha.
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        elif image.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported channel count: {image.shape[2]}")
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # Convert to float32 tensor, robust to either uint8 [0..255] or float [0..1].
    img = image.astype(np.float32)
    max_val = float(np.max(img)) if img.size else 0.0
    if max_val > 1.5:
        img = img / 255.0

    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # Resize to the network input size used during training.
    img_tensor = F.interpolate(
        img_tensor, size=(256, 256), mode="bilinear", align_corners=False
    )

    with torch.no_grad():
        logits_sum = None
        for net in models:
            logits = net(img_tensor)
            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum = logits_sum + logits
        avg_logits = logits_sum / len(models)
        probs = torch.sigmoid(avg_logits)

    # Threshold to a binary mask.
    threshold = _inference_threshold
    binary = (probs > threshold).to(torch.uint8)

    # Resize back to original image size using nearest-neighbor.
    binary = F.interpolate(
        binary.float(), size=(original_h, original_w), mode="nearest"
    )
    binary = binary.to(torch.uint8).squeeze(0).squeeze(0)

    # Convert to numpy and apply post-processing to clean up predictions.
    mask_np = (binary.cpu().numpy() * 255).astype(np.uint8)
    mask_cleaned = _apply_postprocessing(mask_np)

    return mask_cleaned
