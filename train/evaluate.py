"""
train/evaluate.py
─────────────────
Evaluate a trained model checkpoint and print per-class accuracy + confusion matrix.

Usage:
    python train/evaluate.py --data data/ --weights weights/model.pt
"""

import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from train.dataset import ChessSquareDataset, CLASSES
from train.model import build_model


def evaluate(data_dir: str, weights_path: str, batch_size: int = 128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = ChessSquareDataset(data_dir, split="val")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nPer-class report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Accuracy: {accuracy*100:.2f}%)")
    plt.tight_layout()
    plt.savefig("weights/confusion_matrix.png", dpi=150)
    print("\nConfusion matrix saved to weights/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data")
    parser.add_argument("--weights", default="weights/model.pt")
    parser.add_argument("--batch",   type=int, default=128)
    args = parser.parse_args()
    evaluate(args.data, args.weights, args.batch)
