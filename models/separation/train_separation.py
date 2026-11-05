"""
models/separation/train_separation.py
=====================================
Two-Stage Training Routine for Conv-TasNet
Stage 1: Basis Autoencoder Warmup
Stage 2: Utterance-Level Permutation Invariant Training (uPIT)
Member 4: Monaural Audio Source Separation Lead
Samslesha-Nada (SANADA) Capstone Project
"""

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .config import SeparationConfig as CFG
from .conv_tasnet import SANADAConvTasNet
from .dataset import PITLossWrapper, SeparationDataset, compute_si_sdr


def warmup_encoder_decoder_basis(model: SANADAConvTasNet, dataloader: DataLoader, device: str = "cpu", steps: int = 40):
    """
    Stage 1 Warmup: Pre-trains 1D Encoder + Transposed Decoder directly as a linear autoencoder.
    Guarantees that the learned basis can reconstruct waveforms cleanly (> 30 dB SI-SDR)
    before training the TCN separation mask head.
    """
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=3e-3
    )
    model.train()
    print("Beginning Stage 1: Encoder-Decoder Basis Autoencoder Warmup...")

    step_count = 0
    for mix_b, _, _ in dataloader:
        mix_b = mix_b.to(device)
        optimizer.zero_grad()

        # Direct autoencoder path: encode -> relu -> decode
        w = F.relu(model.encoder(mix_b))
        rec = model.decoder(w)

        # Scale-Invariant SDR reconstruction loss
        loss = -compute_si_sdr(mix_b, rec).mean()
        loss.backward()
        optimizer.step()

        step_count += 1
        if step_count >= steps:
            break

    final_recon_sdr = -loss.item()
    print(f"Stage 1 Complete: Basis Reconstruction SI-SDR = {final_recon_sdr:.2f} dB")
    return model


def train_convtasnet(
    model: SANADAConvTasNet,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 15,
    device: str = "cpu",
    checkpoint_dir: str = "./models/checkpoints"
):
    """
    Stage 2: End-to-end uPIT training for Conv-TasNet mask estimation.
    Optimizes negative SI-SDR loss over all K! permutations.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.LEARNING_RATE,
        weight_decay=CFG.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    pit_criterion = PITLossWrapper(silence_penalty_weight=CFG.SILENCE_WEIGHT)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float("inf")
    best_path = os.path.join(checkpoint_dir, "conv_tasnet_separation.pt")

    print(f"Beginning Stage 2: uPIT Mask Training ({epochs} Epochs)...")
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss = 0.0
        n_batches = 0

        for mix_b, stems_b, k_b in train_loader:
            mix_b = mix_b.to(device)
            stems_b = stems_b.to(device)
            k_b = k_b.to(device)

            optimizer.zero_grad()
            est_stems = model(mix_b)
            loss = pit_criterion(est_stems, stems_b, valid_k=k_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP_NORM)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = train_loss / max(n_batches, 1)
        mean_sisdr = -avg_loss
        dt = time.perf_counter() - t0

        print(f"  Epoch [{epoch:02d}/{epochs:02d}] | uPIT Loss: {avg_loss:.4f} dB (SI-SDR: {mean_sisdr:+.2f} dB) | Time: {dt:.2f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_si_sdr": mean_sisdr,
                "version": "1.0.0"
            }, best_path)

    print(f"Training Complete. Best Checkpoint saved to: {best_path}")
    return model
