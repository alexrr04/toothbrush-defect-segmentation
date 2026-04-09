import csv
import json
import os
import random
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as v2
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ToothbrushSegmentationDataset
from unet import UNet


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = tc.sigmoid(logits)
        probs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class BCEPlusDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def build_criterion(loss_name, pos_weight):
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_name == "bce_dice":
        return BCEPlusDiceLoss(bce_weight=0.5, pos_weight=pos_weight)
    raise ValueError(f"Unsupported loss_name: {loss_name}")


def _set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tc.manual_seed(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed(seed)
        tc.cuda.manual_seed_all(seed)

    # Favor determinism for fair seed-based model comparison.
    tc.backends.cudnn.deterministic = True
    tc.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    worker_seed = tc.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must define a key/value mapping.")
    return data


def _create_run_dir(output_root, run_name):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    if run_name:
        candidate = root / run_name
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idx = 0
    while True:
        suffix = f"_{idx}" if idx else ""
        candidate = root / f"run_{timestamp}{suffix}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        idx += 1


def _get_or_create_run_dir(output_root, run_name, resume):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    if run_name:
        candidate = root / run_name
        if resume:
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Resume requested but run directory does not exist: {candidate}"
                )
            return candidate

        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate

    return _create_run_dir(output_root, run_name)


def _save_run_config(run_dir, config):
    config_json_path = run_dir / "config.json"
    with open(config_json_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)

    config_yaml_path = run_dir / "config.yaml"
    with open(config_yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=True)


def _save_history_csv(history, csv_path):
    fieldnames = ["epoch", "train_loss", "val_loss", "lr", "is_best"]
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _save_training_plot(history, plot_path):
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    run_dir,
    early_stopping_patience,
    resume=False,
    resume_checkpoint_path=None,
    save_improvement_checkpoints=True,
    cleanup_resume_checkpoint_on_success=True,
    seed=None,
):
    """
    Training loop for the U-Net model.

    Args:
        model: The U-Net model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run training on (cpu or cuda)
        epochs: Number of epochs to train
        run_dir: Output run directory
        early_stopping_patience: Early stopping patience in epochs
    """
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_epoch = 0

    best_save_path = run_dir / "best_unet_model.pth"
    last_save_path = run_dir / "last_unet_model.pth"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    default_resume_path = run_dir / "resume_checkpoint.pth"
    if resume_checkpoint_path is not None:
        resume_path = Path(resume_checkpoint_path)
    else:
        resume_path = default_resume_path

    history_csv_path = run_dir / "history.csv"
    summary_path = run_dir / "summary.json"
    loss_plot_path = run_dir / "loss_curve.png"

    if resume:
        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume requested but checkpoint not found: {resume_path}"
            )

        checkpoint = tc.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        history = checkpoint.get("history", [])

        # Restore RNG state to continue run deterministically after interruption.
        python_rng_state = checkpoint.get("python_rng_state")
        numpy_rng_state = checkpoint.get("numpy_rng_state")
        torch_rng_state = checkpoint.get("torch_rng_state")
        cuda_rng_state = checkpoint.get("cuda_rng_state")

        if python_rng_state is not None:
            random.setstate(python_rng_state)
        if numpy_rng_state is not None:
            np.random.set_state(numpy_rng_state)
        if torch_rng_state is not None:
            tc.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None and tc.cuda.is_available():
            tc.cuda.set_rng_state_all(cuda_rng_state)

        print(
            f"Resumed from {resume_path} at epoch {start_epoch}/{epochs}. "
            f"Best val loss so far: {best_val_loss:.6f}"
        )

        # Ensure model/optimizer tensors are on current device.
        model.to(device)

    print("\nStarting Training...")
    for epoch in range(start_epoch, epochs):
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

        # -- Validation Phase --
        model.eval()
        val_loss = 0.0

        val_loop = tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False
        )
        with tc.no_grad():
            for images, masks in val_loop:
                images, masks = images.to(device), masks.to(device)

                predictions = model(images)
                loss = criterion(predictions, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # -- Checkpointing --
        improved = avg_val_loss < best_val_loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            tc.save(model.state_dict(), best_save_path)
            print(f"*** Validation loss improved! Saved model to {best_save_path} ***")
            if save_improvement_checkpoints:
                improved_path = (
                    checkpoint_dir
                    / f"best_epoch_{epoch + 1:04d}_val_{avg_val_loss:.6f}.pth"
                )
                tc.save(model.state_dict(), improved_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    "Early stopping triggered "
                    f"after {early_stopping_patience} epochs without improvement."
                )
                break

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": current_lr,
                "is_best": int(improved),
            }
        )

        tc.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "seed": seed,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": tc.get_rng_state(),
                "cuda_rng_state": tc.cuda.get_rng_state_all()
                if tc.cuda.is_available()
                else None,
            },
            resume_path,
        )

    tc.save(model.state_dict(), last_save_path)
    _save_history_csv(history, history_csv_path)
    _save_training_plot(history, loss_plot_path)

    summary = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "seed": seed,
        "best_model_path": str(best_save_path),
        "last_model_path": str(last_save_path),
        "resume_checkpoint_path": str(resume_path),
        "history_csv": str(history_csv_path),
        "loss_curve_png": str(loss_plot_path),
    }
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    if cleanup_resume_checkpoint_on_success and resume_path.exists():
        os.remove(resume_path)
        print(f"Removed temporary resume checkpoint: {resume_path}")

    print("\nTraining Complete!")
    print(f"Run outputs saved in: {run_dir}")
    print(f"Best model weights are saved at: {best_save_path}")


def main():
    print("Initializing Training Pipeline...")

    parser = ArgumentParser(
        description="Train U-Net for toothbrush defect segmentation"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--loss", choices=["bce", "bce_dice"], default=None)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="If set, overrides auto-computed positive class weight.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to resume checkpoint (.pth with optimizer/scheduler/history state).",
    )
    args = parser.parse_args()

    default_config = {
        "dataset_root": "data/toothbrush_dataset",
        "batch_size": 8,
        "epochs": 400,
        "learning_rate": 1e-4,
        "early_stopping_patience": 40,
        "loss_name": "bce_dice",
        "output_root": "trained_runs",
        "run_name": None,
        "pos_weight": None,
        "seed": 42,
        "resume": False,
        "resume_checkpoint": None,
        "save_improvement_checkpoints": True,
        "cleanup_resume_checkpoint_on_success": True,
    }

    file_config = {}
    if args.config is not None:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        file_config = _load_yaml_config(args.config)

    config = {**default_config, **file_config}
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.patience is not None:
        config["early_stopping_patience"] = args.patience
    if args.seed is not None:
        config["seed"] = args.seed
    if args.loss is not None:
        config["loss_name"] = args.loss
    if args.pos_weight is not None:
        config["pos_weight"] = args.pos_weight
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.output_root is not None:
        config["output_root"] = args.output_root
    if args.resume:
        config["resume"] = True
    if args.resume_checkpoint is not None:
        config["resume_checkpoint"] = args.resume_checkpoint

    # --- 1. Configurations & Hyperparameters ---
    dataset_root = config["dataset_root"]
    train_csv = os.path.join(dataset_root, "training.csv")
    val_csv = os.path.join(dataset_root, "validation.csv")
    test_csv = os.path.join(dataset_root, "testing.csv")
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    learning_rate = float(config["learning_rate"])
    early_stopping_patience = int(config["early_stopping_patience"])
    seed = int(config["seed"])
    loss_name = config["loss_name"]

    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    _set_global_seed(seed)
    print(f"Using seed: {seed}")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing {train_csv}. Run prepare_data.py first.")
    if not os.path.exists(val_csv):
        print("validation.csv not found. Falling back to testing.csv for validation.")
        val_csv = test_csv
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Missing {test_csv}. Run prepare_data.py first.")

    run_name = config.get("run_name")
    if run_name is None:
        run_name = f"{loss_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = _get_or_create_run_dir(
        config["output_root"], run_name, bool(config.get("resume", False))
    )

    config["resolved_train_csv"] = train_csv
    config["resolved_val_csv"] = val_csv
    config["resolved_test_csv"] = test_csv
    config["resolved_run_dir"] = str(run_dir)
    config["device"] = str(device)
    _save_run_config(run_dir, config)

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
    val_dataset = ToothbrushSegmentationDataset(val_csv, transforms=test_transforms)
    test_dataset = ToothbrushSegmentationDataset(test_csv, transforms=test_transforms)

    data_loader_generator = tc.Generator()
    data_loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=_seed_worker,
        generator=data_loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=_seed_worker,
        generator=data_loader_generator,
    )
    print(
        f"Loaded {len(train_dataset)} training, {len(val_dataset)} validation, "
        f"and {len(test_dataset)} testing images."
    )

    # --- 4. Model, Loss, and Optimizer ---
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Estimate class balance from training split.
    defect_count = int(train_dataset.dataframe["is_defective"].sum())
    good_count = int(len(train_dataset) - defect_count)
    computed_pos_weight = good_count / max(defect_count, 1)
    if config["pos_weight"] is not None:
        computed_pos_weight = float(config["pos_weight"])
    pos_weight = tc.tensor([computed_pos_weight], device=device)

    criterion = build_criterion(loss_name=loss_name, pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=12,
        min_lr=1e-6,
    )

    print(f"Using loss: {loss_name}")
    print(f"Computed pos_weight: {computed_pos_weight:.4f}")

    # --- 5. Training ---
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        run_dir=run_dir,
        early_stopping_patience=early_stopping_patience,
        resume=bool(config.get("resume", False)),
        resume_checkpoint_path=config.get("resume_checkpoint"),
        save_improvement_checkpoints=bool(
            config.get("save_improvement_checkpoints", True)
        ),
        cleanup_resume_checkpoint_on_success=bool(
            config.get("cleanup_resume_checkpoint_on_success", True)
        ),
        seed=seed,
    )


if __name__ == "__main__":
    main()
