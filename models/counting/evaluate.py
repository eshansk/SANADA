"""
models/counting/evaluate.py
===========================
Formal Evaluation Metric Suite for Active Source Counting
Member 2: Audio Preprocessing & Deep Learning Source Counting Lead

Metrics Computed:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - Exact Accuracy: % where round(clamp(y_pred, 1)) == y_true
  - Relaxed Accuracy: % where |round(clamp(y_pred, 1)) - y_true| <= 1
"""

import torch
import numpy as np


def compute_counting_metrics(preds: torch.Tensor, targets: torch.Tensor):
    """
    Computes MAE, RMSE, Exact Accuracy, and Relaxed Accuracy (+-1).
    """
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds, dtype=torch.float32)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32)

    preds_cont = preds.detach().cpu()
    targets = targets.detach().cpu()

    mae = torch.mean(torch.abs(preds_cont - targets)).item()
    rmse = torch.sqrt(torch.mean((preds_cont - targets) ** 2)).item()

    # Discrete integer evaluation: round and clamp to min=1
    k_hat = torch.round(torch.clamp(preds_cont, min=1.0))
    k_true = torch.round(targets)

    exact_acc = (k_hat == k_true).float().mean().item() * 100.0
    relaxed_acc = (torch.abs(k_hat - k_true) <= 1.0).float().mean().item() * 100.0

    return mae, rmse, exact_acc, relaxed_acc
