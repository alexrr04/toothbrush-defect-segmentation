import os

import numpy as np
from PIL import Image

import model


def _load_rgb(path: str) -> np.ndarray:
    img = Image.open(path)
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def main() -> None:
    dataset_root = os.path.join("data", "toothbrush_dataset", "toothbrush", "train")
    candidates = []
    for cls in ("good", "defective"):
        cls_dir = os.path.join(dataset_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for name in sorted(os.listdir(cls_dir)):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                candidates.append(os.path.join(cls_dir, name))
        if len(candidates) >= 5:
            break

    if not candidates:
        raise SystemExit(
            "No images found under data/. Run fetch_data.py + prepare_data.py first."
        )

    for p in candidates[:5]:
        img = _load_rgb(p)
        mask = model.predict(img)

        uniq = np.unique(mask)
        white_ratio = float(np.mean(mask == 255))

        print("-", p)
        print(
            "  img:",
            img.shape,
            img.dtype,
            "mask:",
            mask.shape,
            mask.dtype,
            "unique:",
            uniq[:10],
            "white_ratio:",
            round(white_ratio, 6),
        )

        assert mask.shape == img.shape[:2]
        assert mask.dtype == np.uint8
        assert set(uniq.tolist()).issubset({0, 255})


if __name__ == "__main__":
    main()
