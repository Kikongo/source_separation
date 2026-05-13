import torch
import torch.nn as nn


class SourceSeparationLoss(nn.Module):
    """
    Loss function for source separation.
    Supports both 2 and 4 source separation using weighted L1 loss.
    """

    def __init__(self, n_sources=2, alpha=None, **kwargs):
        """
        Args:
            n_sources (int): number of sources (2 or 4)
            alpha (float or list): weighting for each source
                - if float: apply to first source (vocals), rest equally weighted
                - if list: must have length n_sources
        """
        super().__init__()
        self.n_sources = n_sources
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        # Set alpha weights
        if alpha is None:
            alpha = 0.707 if n_sources == 2 else [0.25, 0.25, 0.25, 0.25]
        
        if isinstance(alpha, (int, float)):
            if n_sources == 2:
                self.alpha = [alpha, 1 - alpha]
            else:
                # Distribute weight: first source gets alpha, rest share equally
                remaining = 1 - alpha
                self.alpha = [alpha] + [remaining / (n_sources - 1)] * (n_sources - 1)
        else:
            self.alpha = list(alpha)
            assert len(self.alpha) == n_sources, f"Alpha must have {n_sources} elements"

    def forward(self, predicted_sources, true_sources, **kwargs):
        """
        Compute weighted L1 loss for source separation.

        Args:
            predicted_sources: [B, n_sources, F, T]
            true_sources: [B, n_sources, F, T]

        Returns:
            dict with 'loss' key containing the total loss
        """
        if predicted_sources.shape[1] != self.n_sources:
            raise ValueError(
                f"Expected {self.n_sources} sources, got {predicted_sources.shape[1]}"
            )

        total_loss = 0.0
        source_losses = {}

        for source_idx in range(self.n_sources):
            pred = predicted_sources[:, source_idx].unsqueeze(1)
            true = true_sources[:, source_idx].unsqueeze(1)
            
            # Use L1 loss for each source
            source_loss = self.l1_loss(pred, true)
            source_losses[f"loss_source_{source_idx}"] = source_loss
            
            # Add weighted loss
            total_loss = total_loss + self.alpha[source_idx] * source_loss

        source_losses["loss"] = total_loss
        return source_losses


class SourceSeparationLossWithConsistency(nn.Module):
    """
    Loss function combining L1 loss with consistency regularization.
    """

    def __init__(self, n_sources=2, alpha=None, consistency_weight=0.1, **kwargs):
        """
        Args:
            n_sources (int): number of sources (2 or 4)
            alpha (float or list): weighting for each source
            consistency_weight (float): weight for consistency loss
        """
        super().__init__()
        self.n_sources = n_sources
        self.consistency_weight = consistency_weight
        self.base_loss = SourceSeparationLoss(n_sources=n_sources, alpha=alpha)
        self.l1_loss = nn.L1Loss()

    def forward(self, predicted_sources, true_sources, **kwargs):
        """
        Compute loss with consistency regularization.
        Consistency: predicted sources should sum to approximately the mixture.
        """
        base_losses = self.base_loss(predicted_sources, true_sources)
        base_total_loss = base_losses["loss"]

        # Consistency loss: sum of predicted sources should be close to mixture
        mixture = kwargs.get("mixture", None)
        if mixture is not None and mixture.shape[1] == 1:
            predicted_mixture = predicted_sources.sum(dim=1, keepdim=True)
            consistency_loss = self.l1_loss(predicted_mixture, mixture)
            base_losses["consistency_loss"] = consistency_loss
            total_loss = base_total_loss + self.consistency_weight * consistency_loss
        else:
            total_loss = base_total_loss

        base_losses["loss"] = total_loss
        return base_losses
