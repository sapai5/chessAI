"""
train/train.py
──────────────
Training loop for the 13-class chess piece CNN.

Two-stage training:
  Stage 1 (5 epochs):  Backbone frozen, only train the classifier head.
  Stage 2 (15 epochs): Unfreeze all layers, fine-tune end-to-end.

Uses WeightedRandomSampler to balance the heavily skewed class distribution
(empty squares >> rare pieces like queens).

Usage:
    python -m train.train --data data/ --epochs 20 --batch 128 --out weights/model.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from train.dataset import ChessSquareDataset, make_balanced_loader, make_val_loader
from train.model import build_model, freeze_backbone, unfreeze_all


# ── Training helpers ─────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        pbar = tqdm(loader, desc="Training" if training else "Validating", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            pbar.set_postfix(acc=f"{correct/total:.4f}", loss=f"{total_loss/total:.4f}")

    return total_loss / total, correct / total


# ── Main ─────────────────────────────────────────────────────────────────────

def train(data_dir: str, out_path: str, epochs: int, batch_size: int,
          max_per_class: int = 20_000, resume_from: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Datasets
    print("\nBuilding datasets (max_per_class={:,})...".format(max_per_class))
    train_ds = ChessSquareDataset(data_dir, split="train", max_per_class=max_per_class)
    val_ds   = ChessSquareDataset(data_dir, split="val",   max_per_class=max_per_class)

    # Balanced train loader + standard val loader
    train_loader = make_balanced_loader(train_ds, batch_size=batch_size, num_workers=0)
    val_loader   = make_val_loader(val_ds, batch_size=256, num_workers=0)

    # Model
    model = build_model(pretrained=True).to(device)

    # --- Resume / Fine-tune mode ---
    if resume_from and Path(resume_from).exists():
        print(f"\nLoading existing weights from: {resume_from}")
        state = torch.load(resume_from, map_location=device)
        model.load_state_dict(state)
        print("  Weights loaded. Fine-tuning on new data (skipping Stage 1).")
        fine_tune_only = True
    else:
        fine_tune_only = False

    criterion = nn.CrossEntropyLoss()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')

    # ── Stage 1: frozen backbone ──────────────────────────────────────────────
    if fine_tune_only:
        print("\n-- Skipping Stage 1 (resuming from existing weights) --")
    else:
        STAGE1 = min(2 if epochs <= 5 else 5, epochs)
        print(f"\n-- Stage 1: frozen backbone ({STAGE1} epochs) --")
        freeze_backbone(model)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=3e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=STAGE1, eta_min=1e-5)

        for epoch in range(1, STAGE1 + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
            vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,      device, False)
            scheduler.step()
            print(f"  Epoch {epoch:02d}/{STAGE1}  "
                  f"train={tr_acc:.4f}  val={vl_acc:.4f}  "
                  f"loss={tr_loss:.4f}  ({time.time()-t0:.0f}s)")
            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                torch.save(model.state_dict(), out_path)
                print(f"    -> Saved (best val_loss={best_val_loss:.4f})")
        STAGE1 = 0  # so STAGE2 = epochs - 0 = full epochs

    STAGE1 = 0 if fine_tune_only else min(2 if epochs <= 5 else 5, epochs)

    # ── Stage 2: full fine-tune ───────────────────────────────────────────────
    STAGE2 = epochs - STAGE1
    if STAGE2 > 0:
        print(f"\n-- Stage 2: full fine-tune ({STAGE2} epochs) --")
        unfreeze_all(model)
        # Use a lower LR when resuming — backbone weights are already good
        lr = 3e-5 if fine_tune_only else 1e-4
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=STAGE2, eta_min=1e-6)

        for epoch in range(1, STAGE2 + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
            vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,      device, False)
            scheduler.step()
            print(f"  Epoch {epoch:02d}/{STAGE2}  "
                  f"train={tr_acc:.4f}  val={vl_acc:.4f}  "
                  f"loss={tr_loss:.4f}  ({time.time()-t0:.0f}s)")
            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                torch.save(model.state_dict(), out_path)
                print(f"    -> Saved (best val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",          default="data",             help="Dataset root")
    parser.add_argument("--out",           default="weights/model.pt", help="Output weights path")
    parser.add_argument("--epochs",        type=int, default=20)
    parser.add_argument("--batch",         type=int, default=128)
    parser.add_argument("--max-per-class", type=int, default=20_000,
                        help="Cap each class at N samples (handles class imbalance)")
    parser.add_argument("--resume",        default=None,
                        help="Path to existing .pt weights to fine-tune from (skips Stage 1)")
    args = parser.parse_args()
    train(args.data, args.out, args.epochs, args.batch, args.max_per_class, resume_from=args.resume)
