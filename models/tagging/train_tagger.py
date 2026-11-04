"""
models/tagging/train_tagger.py
==============================
Training Script for Member 3: Conformer Backbone + Asymmetric Loss (ASL).
Paper Reference: Section 5.2, Table 5 of SANADA Base Paper.
"""

import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.tagging.config import TaggingConfig as CFG
from models.tagging.conformer_tagger import SANADAConformerTagger
from models.tagging.asl_loss import AsymmetricLoss
from models.tagging.dataset import CachedTaggingDataset


def train_conformer_tagger(epochs: int = 3, batch_size: int = 16, save_dir: str = "./checkpoints"):
    print("=" * 70)
    print("TRAINING SANADA CONFORMER TAGGER WITH ASYMMETRIC LOSS (ASL)")
    print(f"ASL Params: gamma_neg={CFG.GAMMA_NEG}, gamma_pos={CFG.GAMMA_POS}, clip={CFG.CLIP_MARGIN}")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CFG.SEED)

    dataset = CachedTaggingDataset(num_samples=320, num_classes=CFG.NUM_CLASSES, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SANADAConformerTagger(num_classes=CFG.NUM_CLASSES).to(device)
    criterion = AsymmetricLoss(
        gamma_neg=CFG.GAMMA_NEG,
        gamma_pos=CFG.GAMMA_POS,
        clip=CFG.CLIP_MARGIN
    )
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)

    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for mels, labels in loader:
            mels, labels = mels.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(mels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} ({elapsed:.2f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(save_dir, "conformer_tagger_best.pt")
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nTraining Complete. Best Checkpoint saved to: {ckpt_path}")
    return model


if __name__ == "__main__":
    train_conformer_tagger()
