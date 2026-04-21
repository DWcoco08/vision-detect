"""ResNet18-based severity estimation model for damage crops."""

import logging

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard transform for severity model input
SEVERITY_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class SeverityNet(nn.Module):
    """ResNet18 backbone with regression head for severity scoring.

    Outputs a single value in range [0, 100] representing
    damage severity percentage.
    """

    def __init__(self):
        super().__init__()
        # Load pretrained ResNet18 backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features

        # Replace classification head with regression head
        backbone.fc = nn.Linear(in_features, 1)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, 224, 224).

        Returns:
            Severity scores (B,) in range [0, 100].
        """
        out = self.backbone(x)  # (B, 1)
        out = torch.sigmoid(out) * 100.0  # Clamp to [0, 100]
        return out.squeeze(-1)  # (B,)


class SeverityPredictor:
    """Inference wrapper for severity estimation.

    Loads trained SeverityNet weights and predicts severity
    score for a single damage crop image.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """Initialize predictor with trained weights.

        Args:
            model_path: Path to severity model .pth file.
            device: Torch device ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.model = SeverityNet()

        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
        except FileNotFoundError:
            logger.warning(
                "Severity model not found: %s. "
                "Predictions will return -1.0. Train a model first.",
                model_path,
            )
            self._loaded = False

    def predict(self, image: np.ndarray) -> float:
        """Predict severity score for a damage crop.

        Args:
            image: Cropped damage region as RGB numpy array (H, W, 3).

        Returns:
            Severity score 0-100, or -1.0 if model not loaded.
        """
        if not self._loaded:
            return -1.0

        # Preprocess: resize, normalize, batch
        tensor = SEVERITY_TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            score = self.model(tensor).item()

        return round(score, 1)
