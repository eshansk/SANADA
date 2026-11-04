"""
models/counting/train_counter.py
================================
Training Loop with Smooth L1 (Huber) Loss and Cosine Annealing
Member 2: Audio Preprocessing & Deep Learning Source Counting Lead
"""

import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .config import CountingConfig as CFG
from .conformer_counter import SANADAConformerCounter
from .evaluate import compute_counting_metrics


def train_conformer_counter(train_loader, val_loader, num_epochs: int = 5, device: str = "cpu"):
    """
    Trains the 4-Block Conformer regressor using Smooth L1 (Huber) Loss.
    """
    model = SANADAConformerCounter().to(device)
    optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss(beta=CFG.HUBER_BETA)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    history = []
    print(f"Starting Conformer Counter Training ({num_epochs} Epochs on {device})...")

    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss = 0.0

        for mel, targets in train_loader:
            mel, targets = mel.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(mel)
            loss = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for mel, targets in val_loader:
                mel, targets = mel.to(device), targets.to(device)
                preds = model(mel)
                val_preds.append(preds)
                val_targets.append(targets)

        if val_preds:
            cat_preds = torch.cat(val_preds)
            cat_targets = torch.cat(val_targets)
            mae, rmse, exact, relaxed = compute_counting_metrics(cat_preds, cat_targets)
        else:
            mae, rmse, exact, relaxed = 0.0, 0.0, 0.0, 0.0

        epoch_time = time.perf_counter() - t0
        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} | Val MAE: {mae:.4f} | Exact: {exact:.1f}% | Time: {epoch_time:.2f}s")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": mae,
            "exact_acc": exact,
            "relaxed_acc": relaxed
        })

    return model, history
