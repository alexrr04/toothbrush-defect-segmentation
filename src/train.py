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
        test_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on (cpu or cuda)
        epochs: Number of epochs to train
        weights_dir: Directory to save model weights
    """
    best_val_loss = float("inf")
    save_path = os.path.join(weights_dir, "best_unet_model.pth")

    print("\nStarting Training...")
    for epoch in range(epochs):
        # -- Training Phase --
        model.train()
        train_loss = 0.0

        # tqdm adds a nice progress bar to your terminal/notebook
        train_loop = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False
        )
        for images, masks in train_loop:
            images, masks = images.to(device), masks.to(device)

            # 1. Forward pass
            predictions = model(images)
            loss = criterion(predictions, masks)

            # 2. Backward pass & optimize
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()  # Compute new gradients
            optimizer.step()  # Update weights

            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # -- Validation Phase --
        model.eval()
        val_loss = 0.0

        val_loop = tqdm(
            test_loader, desc=f"Epoch {epoch + 1}/{epochs} [Test]", leave=False
        )
        with tc.no_grad():
            for images, masks in val_loop:
                images, masks = images.to(device), masks.to(device)

                predictions = model(images)
                loss = criterion(predictions, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # -- Checkpointing --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            tc.save(model.state_dict(), save_path)
            print(f"*** Validation loss improved! Saved model to {save_path} ***")

    print("\nTraining Complete!")
    print(f"Best model weights are saved at: {save_path}")


def main():
    print("Initializing Training Pipeline...")

    # --- 1. Configurations & Hyperparameters ---
    dataset_root = "data/toothbrush_dataset"
    train_csv = os.path.join(dataset_root, "training.csv")
    test_csv = os.path.join(dataset_root, "testing.csv")
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)

    batch_size = 16
    epochs = 3  # Quick testing
    learning_rate = 1e-4

    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Data Transforms ---
    train_transforms = v2.Compose(
        [
            v2.Resize((256, 256), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),  # Get more variety
            v2.RandomVerticalFlip(p=0.5),  # Get more variety
            v2.ColorJitter(brightness=0.2, contrast=0.2),  # Prevent overfitting
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

    # num_workers=2 speeds up data loading. Set to 0 if running on Windows causes multiprocessing errors.
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
    # Instantiate our custom U-Net from unet.py
    model = UNet(in_channels=3, out_channels=1).to(device)

    # BCEWithLogitsLoss combines a Sigmoid layer and Binary Cross Entropy.
    # It expects raw logits from the model and target masks of 0s and 1s.
    criterion = nn.BCEWithLogitsLoss()
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
