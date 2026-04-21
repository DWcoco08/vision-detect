"""Training script for severity estimation CNN model.

Trains a ResNet18 regression model on labeled damage crop images.
Dataset: folder of images + CSV with (filename, severity) pairs.

Usage:
    python train_severity.py --images-dir data/severity/images \
                             --labels-csv data/severity/labels.csv \
                             --epochs 20 --batch-size 16
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from models.severity_model import SEVERITY_TRANSFORM, SeverityNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class SeverityDataset(Dataset):
    """Dataset for severity regression training.

    Reads image paths and severity labels from CSV file.
    Each sample is a (transformed_image, severity_score) pair.
    """

    def __init__(self, images_dir: str, labels_csv: str, transform=None):
        """Initialize dataset.

        Args:
            images_dir: Directory containing damage crop images.
            labels_csv: CSV file with columns: filename, severity.
            transform: Torchvision transform pipeline.
        """
        self.images_dir = Path(images_dir)
        self.transform = transform or SEVERITY_TRANSFORM
        self.labels = pd.read_csv(labels_csv)

        # Validate that image files exist
        valid_rows = []
        for _, row in self.labels.iterrows():
            img_path = self.images_dir / row["filename"]
            if img_path.exists():
                valid_rows.append(row)
            else:
                logger.warning("Image not found, skipping: %s", img_path)
        self.labels = pd.DataFrame(valid_rows).reset_index(drop=True)

        logger.info("Dataset loaded: %d samples", len(self.labels))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.labels.iloc[idx]
        img_path = self.images_dir / row["filename"]

        # Load image as RGB
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        tensor = self.transform(image)

        # Severity as float tensor
        severity = torch.tensor(float(row["severity"]), dtype=torch.float32)

        return tensor, severity


def train(args: argparse.Namespace) -> None:
    """Run training loop for severity model.

    Args:
        args: Parsed CLI arguments.
    """
    device = torch.device(args.device)
    logger.info("Training on device: %s", device)

    # Load dataset
    dataset = SeverityDataset(args.images_dir, args.labels_csv)
    if len(dataset) == 0:
        logger.error("No valid samples found. Check images and labels.")
        return

    # Train/val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    logger.info("Train: %d samples | Val: %d samples", train_size, val_size)

    # Initialize model, loss, optimizer
    model = SeverityNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Freeze backbone for first half of training (transfer learning)
    freeze_epochs = args.epochs // 2
    _set_backbone_frozen(model, frozen=True)
    logger.info("Backbone frozen for first %d epochs", freeze_epochs)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs + 1:
            _set_backbone_frozen(model, frozen=False)
            logger.info("Backbone unfrozen for fine-tuning")

        # Training phase
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= train_size

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                predictions = model(images)
                loss = criterion(predictions, targets)
                val_loss += loss.item() * images.size(0)

        val_loss /= max(val_size, 1)

        logger.info(
            "Epoch [%d/%d] Train Loss: %.4f | Val Loss: %.4f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output)
            logger.info("Best model saved to %s (val_loss: %.4f)", args.output, val_loss)

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


def _set_backbone_frozen(model: SeverityNet, frozen: bool) -> None:
    """Freeze or unfreeze ResNet18 backbone layers.

    Args:
        model: SeverityNet model instance.
        frozen: True to freeze, False to unfreeze.
    """
    for name, param in model.backbone.named_parameters():
        if name.startswith("fc"):
            continue  # Always keep fc layer trainable
        param.requires_grad = not frozen


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train severity estimation CNN model"
    )
    parser.add_argument(
        "--images-dir",
        default="data/severity/images",
        help="Directory with damage crop images",
    )
    parser.add_argument(
        "--labels-csv",
        default="data/severity/labels.csv",
        help="CSV file with filename,severity columns",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--output",
        default="weights/severity.pth",
        help="Output model path",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cpu or cuda)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
