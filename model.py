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

from src.unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1)


def _resolve_weights_path() -> str | None:
    """Try to locate the trained weights regardless of current working directory."""
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


_weights_path = _resolve_weights_path()
if _weights_path is not None:
    model.load_state_dict(torch.load(_weights_path, map_location=device))
else:
    print(
        "Warning: Could not find best_unet_model.pth. Predictions will use random weights."
    )

model.to(device)
model.eval()


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
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)

    # Threshold to a binary mask.
    threshold = 0.35
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
