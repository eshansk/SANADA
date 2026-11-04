"""
Asymmetric Loss (ASL) for Multi-Label Sound Event Tagging.
Paper Reference: Ben-Baruch et al., ICCV 2021.
Implemented for SANADA Multi-Label Acoustic Tagging (Member 3).
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Acoustic Tagging.
    
    In multi-label datasets like SANADA FSD-Mix (32 classes), each mixture only
    contains a small fraction of positive classes (e.g., 2 to 4), while 28 to 30
    classes are negative. ASL addresses this severe imbalance by:
      1. Asymmetric focusing: gamma_neg > gamma_pos, down-weighting easy negatives.
      2. Probability shifting (margin clip): discards negatives with p < clip entirely.
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean"
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized network predictions of shape [batch_size, num_classes].
            targets: Binary multi-hot targets of shape [batch_size, num_classes] (0 or 1).
        Returns:
            Computed scalar loss.
        """
        # Compute sigmoid probabilities
        probs = torch.sigmoid(logits)

        # Positive class loss: - y * (1 - p)^gamma_pos * log(p)
        pos_probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
        loss_pos = -targets * torch.pow(1.0 - pos_probs, self.gamma_pos) * torch.log(pos_probs)

        # Negative class probability shifting / margin clipping
        if self.clip is not None and self.clip > 0:
            probs_neg = torch.clamp(probs - self.clip, min=0.0, max=1.0)
        else:
            probs_neg = probs

        # Negative class loss: - (1 - y) * (probs_neg)^gamma_neg * log(1 - probs_neg)
        one_minus_p = torch.clamp(1.0 - probs_neg, min=self.eps, max=1.0)
        loss_neg = -(1.0 - targets) * torch.pow(probs_neg, self.gamma_neg) * torch.log(one_minus_p)

        loss = loss_pos + loss_neg

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
