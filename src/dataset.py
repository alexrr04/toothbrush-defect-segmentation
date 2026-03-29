import pandas as pd
import torch as tc
import torchvision as tv
from torchvision import tv_tensors


class ToothbrushSegmentationDataset(tc.utils.data.Dataset):
    def __init__(self, csv_path, transforms=None):
        self.dataframe = pd.read_csv(csv_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        current_case = self.dataframe.iloc[idx]

        # 1. Load Image
        img_path = current_case["image_path"]
        image = tv.io.read_image(img_path, mode=tv.io.ImageReadMode.RGB)
        _, h, w = image.shape

        image = tv_tensors.Image(image)

        mask_path = current_case["mask_path"]

        if pd.isna(mask_path):
            # Perfect toothbrush: Create a tensor of zeros
            mask = tc.zeros((1, h, w), dtype=tc.uint8)
        else:
            # Defective toothbrush: Load the mask
            mask = tv.io.read_image(mask_path, mode=tv.io.ImageReadMode.GRAY)
            mask = (mask > 127).to(tc.uint8)

        mask = tv_tensors.Mask(mask)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        # Normalize image to [0.0, 1.0] as float32.
        image = image.to(tc.float32) / 255.0
        mask = mask.to(tc.float32)

        return image, mask
