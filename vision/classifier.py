"""
vision/classifier.py
─────────────────────
Load the trained ResNet-18 and classify 64 square crops.

Usage:
    from vision.classifier import PieceClassifier
    clf = PieceClassifier("weights/model.pt")
    labels = clf.predict(crops)   # crops = list of 64 PIL Images
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from train.model import build_model
from train.dataset import CLASSES, IDX_TO_CLASS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class PieceClassifier:
    def __init__(self, weights_path: str):
        self.model = build_model(pretrained=False).to(DEVICE)
        self.model.load_state_dict(
            torch.load(weights_path, map_location=DEVICE)
        )
        self.model.eval()
        print(f"Classifier loaded from {weights_path} on {DEVICE}")

    def predict(self, crops: list[Image.Image]) -> list[str]:
        """
        Classify 64 square crops.

        Args:
            crops: list of PIL Images (from square_slicer.slice_board)

        Returns:
            List of 64 class label strings (e.g., ['empty', 'wR', 'wN', ...])
        """
        tensors = torch.stack([_TRANSFORM(c) for c in crops]).to(DEVICE)

        with torch.no_grad():
            logits = self.model(tensors)           # (64, 13)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().tolist()

        labels = [IDX_TO_CLASS[p] for p in preds]
        return labels

    def predict_with_confidence(self, crops: list[Image.Image]) -> list[tuple[str, float]]:
        """Same as predict() but also returns the confidence score."""
        tensors = torch.stack([_TRANSFORM(c) for c in crops]).to(DEVICE)

        with torch.no_grad():
            logits = self.model(tensors)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = logits.argmax(dim=1).cpu().tolist()
            confs = probs.max(dim=1).values.tolist()

        return [(IDX_TO_CLASS[p], float(c)) for p, c in zip(preds, confs)]
