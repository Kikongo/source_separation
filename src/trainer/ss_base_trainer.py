from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH


class SSBaseTrainer:
    """
    Base class for source separation trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model for source separation.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]).
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based training.
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # configuration to monitor model performance and save best
        self.save_period = self.cfg_trainer.save_period
        self.monitor = self.cfg_trainer.get("monitor", "off")

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        # define checkpoint dir and init everything if required
        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic: Training model for an epoch, evaluating it on
        non-train partitions, and monitoring the performance improvement.
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            # evaluate model performance according to configured metric
            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if stop_process:  # early_stop
                break

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            # log current results
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics

        # Run val/test
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on.
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the evaluation information.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(batch_idx, batch, part)

        return self.evaluation_metrics.result()

    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics.

        Args:
            logs (dict): logs after training and evaluating.
            not_improved_count (int): the current number of epochs without improvement.
        Returns:
            best (bool), stop_process (bool), not_improved_count (int)
        """
        if self.mnt_mode == "off":
            return False, False, 0

        try:
            if self.mnt_mode == "min":
                is_better = logs[self.mnt_metric] <= self.mnt_best
            else:
                is_better = logs[self.mnt_metric] >= self.mnt_best
        except KeyError:
            self.logger.warning(f"Key {self.mnt_metric} is not found. " f"Model performance monitoring is disabled.")
            self.mnt_mode = "off"
            return False, False, 0

        if is_better:
            self.mnt_best = logs[self.mnt_metric]
            not_improved_count = 0
            return True, False, not_improved_count
        else:
            not_improved_count += 1

        if not_improved_count > self.early_stop:
            self.logger.info(f"Validation performance didn't improve for {self.early_stop} epochs. " f"Training stops.")
            return False, True, not_improved_count

        return False, False, not_improved_count

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    def _get_grad_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2.0)
        return total_norm

    def _log_scalars(self, metric_tracker):
        for metric_name in metric_tracker.keys():
            scalar_value = metric_tracker.avg(metric_name)
            self.writer.add_scalar(metric_name, scalar_value)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch.
        """
        raise NotImplementedError()

    @abstractmethod
    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss.
        """
        raise NotImplementedError()

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors of the sample into the device.
        """
        for tensor_for_gpu in ["mixture", "sources"]:
            if isinstance(batch.get(tensor_for_gpu), dict):
                batch[tensor_for_gpu] = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch[tensor_for_gpu].items()
                }
            else:
                batch[tensor_for_gpu] = batch.get(tensor_for_gpu, {})
                if isinstance(batch[tensor_for_gpu], torch.Tensor):
                    batch[tensor_for_gpu] = batch[tensor_for_gpu].to(self.device)
        return batch

    def transform_batch(self, batch):
        """Apply batch transforms if they exist."""
        if self.batch_transforms is None:
            return batch
        if isinstance(self.batch_transforms, dict):
            for transform_name, transform in self.batch_transforms.items():
                if transform_name in batch:
                    batch[transform_name] = transform(batch[transform_name])
        return batch

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints at the end of epochs.
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "monitoring_best": self.mnt_best,
            "config": self.config,
        }

        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, filename)
        self.logger.info(f"Saving checkpoint: {filename} ...")

        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info(f"Saving current best: model_best.pth ...")

        if only_best and not save_best:
            if self.checkpoint_dir.glob("checkpoint-epoch*.pth"):
                self.logger.info("Removing previous checkpoints...")
                for f in self.checkpoint_dir.glob("checkpoint-epoch*.pth"):
                    if f.is_file():
                        f.unlink()

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoint.
        """
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitoring_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning("Model type given in config differs from that in checkpoint. " "This may yield an exception while state_dict is being loaded.")

        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint["config"]["optimizer"] != self.config["optimizer"]:
            self.logger.warning("Optimizer given in config differs from that in checkpoint. " "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.lr_scheduler and checkpoint["lr_scheduler"]:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")

    def _from_pretrained(self, pretrained_path):
        """
        Load pretrained model.
        """
        self.logger.info(f"Loading pretrained model from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.logger.info(f"Pretrained model loaded successfully")
