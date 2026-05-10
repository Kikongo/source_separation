from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed

def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, text_encoder, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        text_encoder (CTCTextEncoder): instance of the text encoder
            for the datasets.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        # dataset partition init
        dataset = instantiate(
            config.datasets[dataset_partition], text_encoder=text_encoder
        )  # instance transforms are defined inside

        assert config.dataloader.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms


def get_dataloaders_ss(config):
    """
    Create dataloaders for source separation models using MusDB.
    
    Properly loads data using: musdb.DB → MusDBSigSepStems → MusDB_segments
    for each song, then combines into a single dataset.

    Args:
        config (DictConfig): hydra experiment config containing:
            - n_sources: 2 or 4
            - subsets: ['train', 'test'] or similar
            - dataloader: batch_size, num_workers, etc.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for each partition
    """
    import torch
    from torch.utils.data import ConcatDataset
    import musdb
    
    dataloaders = {}
    n_sources = config.model.n_sources
    
    # Import the correct dataset classes based on n_sources
    if n_sources == 2:
        from src.datasets.musdb_sigsep_2sources import MusDBSigSepStems
        from src.datasets.musdb_segments_2sources import MusDB_segments
    elif n_sources == 4:
        from src.datasets.musdb_sigsep_4sources import MusDBSigSep as MusDBSigSepStems
        from src.datasets.musdb_segments_4sources import MusDB_segments
    else:
        raise ValueError(f"n_sources must be 2 or 4, got {n_sources}")
    
    # Get subsets from config
    subsets = config.get("subsets", ["train", "test"])
    
    for subset in subsets:
        try:
            # Load musdb for this subset
            mus_db = musdb.DB(subsets=subset, download=False)
            
            if len(mus_db) == 0:
                print(f"Warning: MusDB subset '{subset}' is empty, skipping...")
                continue
            
            # Wrap with MusDBSigSepStems to extract spectrograms
            musdb_sigsep = MusDBSigSepStems(mus_db)
            
            # Create MusDB_segments datasets for each song and combine them
            segment_datasets = []
            for song_idx in range(len(musdb_sigsep)):
                song_data = musdb_sigsep[song_idx]
                song_segments = MusDB_segments(song_data)
                segment_datasets.append(song_segments)
            
            # Combine all songs into one dataset
            combined_dataset = ConcatDataset(segment_datasets)
            
            if len(combined_dataset) == 0:
                print(f"Warning: No segments found for subset '{subset}', skipping...")
                continue
            
            assert config.dataloader.batch_size <= len(combined_dataset), (
                f"The batch size ({config.dataloader.batch_size}) cannot "
                f"be larger than the dataset length ({len(combined_dataset)})"
            )
            
            # Create dataloader
            partition_dataloader = instantiate(
                config.dataloader,
                dataset=combined_dataset,
                collate_fn=collate_fn_ss,
                drop_last=(subset == "train"),
                shuffle=(subset == "train"),
                worker_init_fn=set_worker_seed,
            )
            dataloaders[subset] = partition_dataloader
            
            print(f"Loaded subset '{subset}' with {len(combined_dataset)} total segments from {len(musdb_sigsep)} songs")
        
        except Exception as e:
            print(f"Error loading subset '{subset}': {e}")
            continue
    
    return dataloaders


def collate_fn_ss(batch):
    """
    Collate function for source separation datasets.
    Handles (mixture, sources) tuples from datasets.

    Args:
        batch (list): list of (mixture, sources) tuples

    Returns:
        dict: dictionary with 'mixture' and 'sources' keys containing
              stacked tensors
    """
    import torch

    mixtures = []
    sources_list = []

    for item in batch:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            mixture, sources = item
        else:
            raise ValueError(f"Expected (mixture, sources) tuple, got {type(item)}")

        mixtures.append(torch.tensor(mixture, dtype=torch.float32))
        sources_list.append(torch.tensor(sources, dtype=torch.float32))

    return {
        "mixture": torch.stack(mixtures),
        "sources": torch.stack(sources_list),
    }