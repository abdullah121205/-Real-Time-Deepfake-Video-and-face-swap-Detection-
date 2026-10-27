"""
Deep learning models for deepfake and facial forgery classification.
Includes MesoNet (Meso4, MesoInception4) and MobileNetV3 architectures.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import preprocess_face_for_model


class Meso4(nn.Module):
    """
    MesoNet-4 architecture (Afchar et al., IEEE WIFS 2018).
    A compact mesoscopic convolutional network designed specifically for deepfake detection.
    Features ~28,000 parameters, ensuring high real-time throughput on standard CPUs.
    """

    def __init__(self, num_classes: int = 1):
        super(Meso4, self).__init__()
        # Conv block 1
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 2
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 3
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 4
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        # Fully connected layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 7 * 7, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (B, 3, 224, 224) or (B, 3, 256, 256)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        # Adaptive pool ensures 7x7 spatial size regardless of exact input resolution
        x = F.adaptive_avg_pool2d(x, (7, 7))
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out


class InceptionBlock(nn.Module):
    """Inception-style multi-scale feature extractor for MesoInception-4."""

    def __init__(self, in_channels: int, a: int, b: int, c: int, d: int):
        super(InceptionBlock, self).__init__()
        self.b1 = nn.Conv2d(in_channels, a, kernel_size=1)

        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, b, kernel_size=1),
            nn.Conv2d(b, b, kernel_size=3, padding=1)
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=1),
            nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2)
        )

        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=1),
            nn.Conv2d(d, d, kernel_size=3, padding=3, dilation=3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], dim=1)


class MesoInception4(nn.Module):
    """MesoInception-4 architecture with multi-scale dilation for forgery artifacts."""

    def __init__(self, num_classes: int = 1):
        super(MesoInception4, self).__init__()
        self.inc1 = InceptionBlock(3, 1, 4, 4, 2)
        self.bn1 = nn.BatchNorm2d(11)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inc2 = InceptionBlock(11, 2, 4, 4, 2)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(12, 16, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 7 * 7, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.inc1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.inc2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv1(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv2(x)))
        x = self.pool4(x)

        x = F.adaptive_avg_pool2d(x, (7, 7))
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out


class MobileNetDeepfake(nn.Module):
    """
    Transfer learning wrapper based on torchvision's MobileNetV3-Small.
    Offers modern inverted bottleneck architecture with fast inference.
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = False):
        super(MobileNetDeepfake, self).__init__()
        import torchvision.models as models
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)

        self.features = base.features
        self.avgpool = base.avgpool

        # Custom binary classification head with dropout regularization
        in_features = base.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DeepfakeDetector:
    """
    Main inference controller for deepfake classification.
    Handles device allocation, image preprocessing, batched inference, and weight management.
    """

    def __init__(
        self,
        model_name: str = "meso4",
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            model_name: Architecture type: 'meso4', 'meso_inception', or 'mobilenet'.
            weights_path: Path to PyTorch .pth weight file (optional).
            device: 'cuda', 'cpu', or None (auto-select).
            input_size: Target image dimensions (width, height).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = model_name.lower()
        self.input_size = input_size
        self.weights_path = weights_path

        # Initialize network architecture
        self.model = self._create_model(self.model_name)
        self.model.to(self.device)

        # Load weights if available
        if weights_path and Path(weights_path).is_file():
            self.load_weights(weights_path)
        else:
            self.model.eval()

    def _create_model(self, name: str) -> nn.Module:
        if name in ("meso4", "mesonet"):
            return Meso4(num_classes=1)
        elif name in ("meso_inception", "meso_inc"):
            return MesoInception4(num_classes=1)
        elif name in ("mobilenet", "mobilenet_v3"):
            return MobileNetDeepfake(num_classes=1, pretrained=False)
        else:
            raise ValueError(f"Unsupported model architecture: {name}. Choose from meso4, meso_inception, mobilenet.")

    def load_weights(self, path: str):
        """Loads model weights from a .pth file."""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        print(f"[DeepfakeDetector] Loaded weights from {path}")

    def save_weights(self, path: str):
        """Saves current model weights to a .pth file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[DeepfakeDetector] Model weights saved to {path}")

    @torch.no_grad()
    def predict(self, face_bgr: np.ndarray) -> Tuple[float, str]:
        """
        Runs model inference on a single cropped face image.

        Args:
            face_bgr: Cropped BGR face region (numpy array).

        Returns:
            fake_score: Probability score in [0.0, 1.0] where 1.0 = manipulated/deepfake.
            label: Text classification ('AUTHENTIC' or 'DEEPFAKE').
        """
        if face_bgr is None or face_bgr.size == 0:
            return 0.0, "AUTHENTIC"

        tensor_arr = preprocess_face_for_model(
            face_bgr,
            target_size=self.input_size,
            normalize_imagenet=(self.model_name == "mobilenet")
        )
        input_tensor = torch.from_numpy(tensor_arr).unsqueeze(0).to(self.device)

        self.model.eval()
        logits = self.model(input_tensor)
        prob = torch.sigmoid(logits).item()

        label = "DEEPFAKE" if prob >= 0.50 else "AUTHENTIC"
        return float(prob), label

    @torch.no_grad()
    def predict_batch(self, face_crops: List[np.ndarray]) -> List[Tuple[float, str]]:
        """Batched inference for frames containing multiple faces."""
        if not face_crops:
            return []

        tensors = [
            preprocess_face_for_model(
                crop,
                target_size=self.input_size,
                normalize_imagenet=(self.model_name == "mobilenet")
            )
            for crop in face_crops if crop is not None and crop.size > 0
        ]

        if not tensors:
            return []

        batch_arr = np.stack(tensors, axis=0)
        batch_tensor = torch.from_numpy(batch_arr).to(self.device)

        self.model.eval()
        logits = self.model(batch_tensor)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist()

        if isinstance(probs, float):
            probs = [probs]

        results = [(p, "DEEPFAKE" if p >= 0.50 else "AUTHENTIC") for p in probs]
        return results
