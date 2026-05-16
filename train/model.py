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
import timm

NUM_CLASSES = 13


def build_model(pretrained: bool = True) -> nn.Module:
    """
    Load ResNet-18 from timm with ImageNet pre-trained weights,
    replace the final fully-connected layer with a 13-class head.
    """
    model = timm.create_model(
        "resnet18",
        pretrained=pretrained,
        num_classes=NUM_CLASSES,
    )
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
