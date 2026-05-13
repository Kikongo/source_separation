import numpy as np
import torch
import torch.nn as nn

from src.metrics.tracker import MetricTracker
from src.trainer.mss_base_trainer import MSSBaseTrainer


class MSSTrainer(MSSBaseTrainer):
    """
    Trainer class for source separation models.
    Defines the logic of batch logging and processing for audio source separation tasks.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        Args:
            batch (dict): dict-based batch containing mixture and sources.
                - mixture: [B, 1, F, T] spectrogram of the mixture
                - sources: [B, n_sources, F, T] spectrograms of the sources
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics.
        Returns:
            batch (dict): dict-based batch containing the data, model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        # Forward pass
        mixture = batch["mixture"]
        true_sources = batch["sources"]

        # Ensure mixture has shape [B, 1, F, T]
        if mixture.dim() == 3:
            mixture = mixture.unsqueeze(1)

        # Model prediction
        predicted_sources = self.model(mixture)  # [B, n_sources, F, T]
        batch["predicted_sources"] = predicted_sources

        # Compute loss
        all_losses = self.criterion(predicted_sources, true_sources)
        if isinstance(all_losses, dict):
            batch.update(all_losses)
            loss = all_losses.get("loss", sum(all_losses.values()))
        else:
            batch["loss"] = all_losses
            loss = all_losses

        if self.is_train:
            loss.backward()
            #self._clip_grad_norm()
            self.optimizer.step()
            # if self.lr_scheduler is not None:
            #     self.lr_scheduler.step()

        # Update metrics for each loss
        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Logs spectrograms and audio information.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through process_batch.
            mode (str): train or inference.
        """
        if mode == "train":
            self.log_spectrogram(**batch)
        else:
            self.log_spectrogram(**batch)
            self.log_source_predictions(**batch)

    def log_spectrogram(self, mixture, predicted_sources, sources, **batch):
        """Log input mixture and predicted sources spectrograms."""
        try:
            # Log mixture
            mixture_for_plot = mixture[0, 0].detach().cpu().numpy()
            self.writer.add_image(
                "mixture_spectrogram",
                self._spectrogram_to_image(mixture_for_plot),
            )

            # Log first predicted source
            if predicted_sources is not None:
                pred_source = predicted_sources[0, 0].detach().cpu().numpy()
                self.writer.add_image(
                    "predicted_source_0",
                    self._spectrogram_to_image(pred_source),
                )

            # Log ground truth first source
            true_source = sources[0, 0].detach().cpu().numpy()
            self.writer.add_image(
                "ground_truth_source_0",
                self._spectrogram_to_image(true_source),
            )
        except Exception as e:
            self.logger.warning(f"Could not log spectrogram: {e}")

    def log_source_predictions(self, predicted_sources, sources, **batch):
        """Log source separation quality metrics."""
        try:
            if predicted_sources is None or sources is None:
                return

            # Compute source-wise metrics
            for source_idx in range(sources.shape[1]):
                pred = predicted_sources[:, source_idx].detach().cpu().numpy()
                true = sources[:, source_idx].detach().cpu().numpy()

                # Compute MSE per source
                mse = np.mean((pred - true) ** 2)
                self.writer.add_scalar(f"mse_source_{source_idx}", mse)

                # Compute correlation
                if true.std() > 0 and pred.std() > 0:
                    correlation = np.corrcoef(pred.flatten(), true.flatten())[0, 1]
                    self.writer.add_scalar(f"correlation_source_{source_idx}", correlation)
        except Exception as e:
            self.logger.warning(f"Could not log source predictions: {e}")

    @staticmethod
    def _spectrogram_to_image(spec):
        """Convert spectrogram to normalized image format for logging."""
        spec = np.abs(spec)
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        spec = np.stack([spec] * 3, axis=0)  # Convert to RGB
        return spec

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors into the device.
        """
        if "mixture" in batch and isinstance(batch["mixture"], torch.Tensor):
            batch["mixture"] = batch["mixture"].to(self.device)
        if "sources" in batch and isinstance(batch["sources"], torch.Tensor):
            batch["sources"] = batch["sources"].to(self.device)
        return batch
