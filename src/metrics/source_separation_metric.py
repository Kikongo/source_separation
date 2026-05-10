import numpy as np
import torch

from src.metrics.base_metric import BaseMetric


class SourceSeparationMetric(BaseMetric):
    """
    Metric for evaluating source separation quality.
    Computes average correlation between predicted and ground truth sources.
    """

    def __init__(self, name="ss_correlation", **kwargs):
        super().__init__(name=name, **kwargs)

    def __call__(self, predicted_sources, sources, **batch):
        """
        Compute source separation metrics.

        Args:
            predicted_sources: [B, n_sources, F, T] - predicted source spectrograms
            sources: [B, n_sources, F, T] - ground truth source spectrograms

        Returns:
            float: average correlation across all sources and batch
        """
        if predicted_sources is None or sources is None:
            return 0.0

        predicted_sources = predicted_sources.detach().cpu().numpy()
        sources = sources.detach().cpu().numpy()

        correlations = []
        for batch_idx in range(sources.shape[0]):
            for source_idx in range(sources.shape[1]):
                pred = predicted_sources[batch_idx, source_idx].flatten()
                true = sources[batch_idx, source_idx].flatten()

                # Compute correlation
                if true.std() > 1e-8 and pred.std() > 1e-8:
                    correlation = np.corrcoef(pred, true)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(correlation)

        if correlations:
            return float(np.mean(correlations))
        return 0.0


class SourceSeparationL1Metric(BaseMetric):
    """
    Metric for evaluating source separation using L1 error.
    """

    def __init__(self, name="ss_l1_error", **kwargs):
        super().__init__(name=name, **kwargs)

    def __call__(self, predicted_sources, sources, **batch):
        """
        Compute L1 error between predicted and true sources.

        Args:
            predicted_sources: [B, n_sources, F, T] - predicted source spectrograms
            sources: [B, n_sources, F, T] - ground truth source spectrograms

        Returns:
            float: average L1 error
        """
        if predicted_sources is None or sources is None:
            return 0.0

        l1_error = torch.nn.functional.l1_loss(predicted_sources, sources)
        return float(l1_error.item())


class SourceSeparationMSEMetric(BaseMetric):
    """
    Metric for evaluating source separation using MSE error.
    """

    def __init__(self, name="ss_mse_error", **kwargs):
        super().__init__(name=name, **kwargs)

    def __call__(self, predicted_sources, sources, **batch):
        """
        Compute MSE error between predicted and true sources.

        Args:
            predicted_sources: [B, n_sources, F, T] - predicted source spectrograms
            sources: [B, n_sources, F, T] - ground truth source spectrograms

        Returns:
            float: average MSE error
        """
        if predicted_sources is None or sources is None:
            return 0.0

        mse_error = torch.nn.functional.mse_loss(predicted_sources, sources)
        return float(mse_error.item())
