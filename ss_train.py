import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.trainer.ss_trainer import SSTrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="ss_2sources")
def main(config):
    """
    Main script for training source separation models.
    
    Instantiates the model, optimizer, scheduler, metrics, logger, writer,
    and dataloaders. Runs SSTrainer to train and evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    logger.info(f"Using device: {device}")

    # Build model architecture
    model = instantiate(config.model).to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(model)

    # Get loss function
    loss_function = instantiate(config.loss_function).to(device)
    logger.info(f"Loss function: {loss_function.__class__.__name__}")

    # Setup metrics for training and inference
    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            metrics[metric_type].append(instantiate(metric_config))
            logger.info(f"Added {metric_type} metric: {metrics[metric_type][-1].name}")

    # Setup optimizer and scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    logger.info(f"Optimizer: {optimizer.__class__.__name__}, LR: {config.optimizer.lr}")

    # Setup dataloaders - use the new get_dataloaders_ss function
    from src.datasets.data_utils import get_dataloaders_ss

    logger.info(f"Loading MusDB subsets: {config.subsets}")
    dataloaders = get_dataloaders_ss(config)
    
    if not dataloaders:
        logger.error("No dataloaders were created! Check your MusDB path and subsets.")
        raise RuntimeError("Failed to create dataloaders")
    
    logger.info(
        f"Dataloaders created: {', '.join(f'{k}={len(v)}' for k, v in dataloaders.items())}"
    )

    # Setup trainer
    epoch_len = config.trainer.get("epoch_len")

    trainer = SSTrainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        epoch_len=epoch_len,
        skip_oom=config.trainer.get("skip_oom", True),
        batch_transforms=None,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed.")


if __name__ == "__main__":
    main()

