"""
train/dataset.py
────────────────
PyTorch Dataset for the 13-class chess square classifier.

Loads JPEG crops from:
    data/{class_name}/*.jpg

Usage:
    from train.dataset import ChessSquareDataset, get_transforms, CLASSES
"""

from pathlib import Path
from typing import Literal
import collections
import random

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# ── Class mapping ─────────────────────────────────────────────────────────────

CLASSES = [
    "empty",
    "wP", "wN", "wB", "wR", "wQ", "wK",
    "bP", "bN", "bB", "bR", "bQ", "bK",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: Literal["train", "val"]) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize((64, 64)),
            # NOTE: No horizontal flip — piece orientation matters
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
            transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1),
            transforms.RandomRotation(degrees=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class ChessSquareDataset(Dataset):
    """
    Flat image dataset.  Expects directory structure:
        root/
            empty/  *.jpg
            wP/     *.jpg
            ...

    The dataset has a severe class imbalance (empty >> pieces).
    Use make_balanced_loader() to get a WeightedRandomSampler-backed DataLoader,
    or pass max_per_class to cap overrepresented classes.
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "val"] = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
        max_per_class: int | None = 20_000,
    ):
        """
        Args:
            root:          Dataset root containing class subdirectories.
            split:         'train' or 'val'.
            val_fraction:  Fraction of data to hold out for validation.
            seed:          Random seed for the train/val split.
            max_per_class: Cap each class at this many samples before splitting.
                           Default 20k keeps the dataset manageable while
                           preserving plenty of data for all classes.
        """
        self.root = Path(root)
        self.transform = get_transforms(split)

        rng = random.Random(seed)
        all_samples: list[tuple[Path, int]] = []

        for cls in CLASSES:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            paths = list(cls_dir.glob("*.jpg"))
            rng.shuffle(paths)
            if max_per_class is not None:
                paths = paths[:max_per_class]
            for p in paths:
                all_samples.append((p, CLASS_TO_IDX[cls]))

        rng.shuffle(all_samples)
        split_idx = int(len(all_samples) * (1 - val_fraction))
        self.samples = all_samples[:split_idx] if split == "train" else all_samples[split_idx:]

        # Print summary
        lc = collections.Counter(lbl for _, lbl in self.samples)
        print(f"  [{split}] {len(self.samples):,} samples across {len(lc)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label

    def get_sample_weights(self) -> torch.Tensor:
        """
        Per-sample weights for WeightedRandomSampler.
        Each class gets equal probability of appearing in a batch.
        """
        lc = collections.Counter(lbl for _, lbl in self.samples)
        total = len(self.samples)
        cw = {cls_idx: total / (len(CLASSES) * count) for cls_idx, count in lc.items()}
        return torch.tensor([cw[lbl] for _, lbl in self.samples], dtype=torch.float)


def make_balanced_loader(
    dataset: ChessSquareDataset,
    batch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:
    """
    Build a DataLoader with WeightedRandomSampler for class-balanced training.
    Each epoch draws ~equal samples from each class regardless of raw frequencies.
    """
    weights = dataset.get_sample_weights()
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=True)


def make_val_loader(
    dataset: ChessSquareDataset,
    batch_size: int = 256,
    num_workers: int = 4,
) -> DataLoader:
    """Standard sequential val loader (no sampling)."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
