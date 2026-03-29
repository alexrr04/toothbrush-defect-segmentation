import os

import cv2
import numpy as np
import torch

from src.unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1)

weights_path = "weights/best_unet_model.pth"

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
else:
    print(
        f"Warning: Could not find {weights_path}. Predictions will use random weights."
    )

model.to(device)
model.eval()


def predict(image):
    """
    Predict the segmentation mask for a given image.

    Args:
        image: numpy array, uint8. Shape (H, W, 3) RGB or (H, W) grayscale.

    Returns:
        Binary mask as numpy array of shape (H, W), uint8 with values 0 or 255.
    """
    original_h, original_w = image.shape[:2]

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    input_size = (256, 256)
    resized_img = cv2.resize(image, input_size)

    img_tensor = torch.from_numpy(resized_img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        logits = model(img_tensor)

        # Convert logits into probabilities (0.0 to 1.0)
        probs = torch.sigmoid(logits)

    probs_np = probs.squeeze().cpu().numpy()

    threshold = 0.5
    binary_mask = np.where(probs_np > threshold, 255, 0).astype(np.uint8)

    final_mask = cv2.resize(
        binary_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST
    )

    return final_mask
