import os

import torch as tc
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ToothbrushSegmentationDataset
from unet import UNet


def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    epochs,
    weights_dir,
):
    """
    Training loop for the U-Net model.

    Args:
        model: The U-Net model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on (cpu or cuda)
        epochs: Number of epochs to train
        weights_dir: Directory to save model weights
    """
    best_test_loss = float("inf")
    save_path = os.path.join(weights_dir, "best_unet_model.pth")

    print("\nStarting Training...")
    for epoch in range(epochs):
        # -- Training Phase --
        model.train()
        train_loss = 0.0

        train_loop = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False
        )
        for images, masks in train_loop:
            images, masks = images.to(device), masks.to(device)

            # 1. Forward pass
            predictions = model(images)
            loss = criterion(predictions, masks)

            # 2. Backward pass & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # -- Testing Phase --
        model.eval()
        test_loss = 0.0

        test_loop = tqdm(
            test_loader, desc=f"Epoch {epoch + 1}/{epochs} [Test]", leave=False
        )
        with tc.no_grad():
            for images, masks in test_loop:
                images, masks = images.to(device), masks.to(device)

                predictions = model(images)
                loss = criterion(predictions, masks)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}"
        )

        # -- Checkpointing --
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            tc.save(model.state_dict(), save_path)
            print(f"*** Testing loss improved! Saved model to {save_path} ***")

    print("\nTraining Complete!")
    print(f"Best model weights are saved at: {save_path}")


def main():
    print("Initializing Training Pipeline...")

    # --- 1. Configurations & Hyperparameters ---
    dataset_root = "data/toothbrush_dataset"
    train_csv = os.path.join(dataset_root, "training.csv")
    test_csv = os.path.join(dataset_root, "testing.csv")
    weights_dir = "trained_models"
    os.makedirs(weights_dir, exist_ok=True)

    batch_size = 8
    epochs = 400
    learning_rate = 1e-4

    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Data Transforms ---
    # Enhanced augmentations to improve model robustness and generalization
    train_transforms = v2.Compose(
        [
            v2.Resize((256, 256), antialias=True),
            v2.RandomApply(
                [v2.RandomRotation(degrees=15)], p=0.5
            ),  # Random rotation ±15°
            v2.RandomApply(
                [v2.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5
            ),  # Random translation
            v2.RandomHorizontalFlip(p=0.5),  # Get more variety
            v2.RandomVerticalFlip(p=0.5),  # Get more variety
            v2.RandomApply(
                [
                    v2.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                    )
                ],
                p=0.7,
            ),  # Enhanced color distortion
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3
            ),  # Slight blur for robustness
        ]
    )

    test_transforms = v2.Compose(
        [
            v2.Resize((256, 256), antialias=True),
        ]
    )

    # --- 3. Datasets & DataLoaders ---
    train_dataset = ToothbrushSegmentationDataset(
        train_csv, transforms=train_transforms
    )
    test_dataset = ToothbrushSegmentationDataset(test_csv, transforms=test_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(
        f"Loaded {len(train_dataset)} training and {len(test_dataset)} testing images."
    )

    # --- 4. Model, Loss, and Optimizer ---
    model = UNet(in_channels=3, out_channels=1).to(device)

    pos_weight = tc.tensor([2.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 5. Training ---
    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        weights_dir=weights_dir,
    )


if __name__ == "__main__":
    main()
