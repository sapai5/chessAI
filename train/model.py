"""
train/model.py
──────────────
ResNet-18 fine-tuned for 13-class chess piece classification.

Usage:
    from train.model import build_model
    model = build_model(pretrained=True)
"""

import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 13


def build_model(pretrained: bool = True) -> nn.Module:
    """
    Load ResNet-18 from torchvision with ImageNet pre-trained weights,
    replace the final fully-connected layer with a 13-class head.
    """
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18()
        
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def freeze_backbone(model: nn.Module):
    """Freeze all layers except the final classifier head."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


if __name__ == "__main__":
    model = build_model(pretrained=False)
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    print(f"Model output shape: {out.shape}")   # (4, 13)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
