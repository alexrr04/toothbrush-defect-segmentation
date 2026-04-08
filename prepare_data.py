import os

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    print("Starting data preparation...")

    base_path = os.path.join("data", "toothbrush_dataset", "toothbrush")

    train_dir = os.path.join(base_path, "train")
    masks_dir = os.path.join(base_path, "ground_truth", "defective")

    if not os.path.exists(train_dir):
        print(f"Error: Could not find the training directory at {train_dir}")
        print("Check if the folder extracted differently!")
        return

    dataset_cases = []

    categories = ["good", "defective"]

    for category in categories:
        category_dir = os.path.join(train_dir, category)

        if not os.path.exists(category_dir):
            continue

        for img_name in os.listdir(category_dir):
            img_path = os.path.join(category_dir, img_name)

            if category == "defective":
                mask_name = img_name.replace(".png", "_mask.png")
                mask_path = os.path.join(masks_dir, mask_name)

                is_defective = 1
            else:
                mask_path = None
                is_defective = 0

            dataset_cases.append((img_path, mask_path, is_defective))

    # Create the master dataframe
    df = pd.DataFrame(
        dataset_cases, columns=["image_path", "mask_path", "is_defective"]
    )

    # Split into Training (70%), Validation (10%), and Testing (20%)
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["is_defective"]
    )
    # 10/80 = 12.5% of the remaining data to end at 10% of full dataset.
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.125,
        random_state=42,
        stratify=train_val_df["is_defective"],
    )

    # Save to CSV in the root data folder
    dataset_root = "data/toothbrush_dataset"
    train_csv_path = os.path.join(dataset_root, "training.csv")
    val_csv_path = os.path.join(dataset_root, "validation.csv")
    test_csv_path = os.path.join(dataset_root, "testing.csv")

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(
        "Success! Saved "
        f"{len(train_df)} training, {len(val_df)} validation, and {len(test_df)} testing cases."
    )


if __name__ == "__main__":
    main()
