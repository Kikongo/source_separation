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


class LazyMusDBSegments:
    """
    Lazy-loading dataset for MusDB segments.
    Loads song spectrograms on-demand to avoid RAM overflow.
    Only keeps one or two songs in memory at a time.
    """

    def __init__(self, musdb_sigsep, n_sources=2):
        """
        Args:
            musdb_sigsep: MusDBSigSepStems or MusDBSigSep dataset
            n_sources: 2 or 4
        """
        import numpy as np
        
        self.musdb_sigsep = musdb_sigsep
        self.n_sources = n_sources
        
        # Import correct segment class
        if n_sources == 2:
            from src.datasets.musdb_segments_2sources import MusDB_segments
        elif n_sources == 4:
            from src.datasets.musdb_segments_4sources import MusDB_segments
        else:
            raise ValueError(f"n_sources must be 2 or 4, got {n_sources}")
        
        self.MusDB_segments = MusDB_segments
        
        # Pre-compute segment counts for each song (only load once)
        self.song_segment_counts = []
        self.segment_to_song_idx = []  # Maps global segment index to song index
        global_idx = 0
        
        for song_idx in range(len(musdb_sigsep)):
            song_data = musdb_sigsep[song_idx]
            n_segments = song_data['mixture_segments_spectrograms'].shape[0]
            self.song_segment_counts.append(n_segments)
            
            # Map all segments from this song to the song index
            for _ in range(n_segments):
                self.segment_to_song_idx.append(song_idx)
            
            global_idx += n_segments
        
        self.total_segments = len(self.segment_to_song_idx)
        print(f"LazyMusDBSegments initialized: {len(musdb_sigsep)} songs, "
              f"{self.total_segments} total segments")

    def __len__(self):
        return self.total_segments

    def __getitem__(self, global_idx):
        """
        Get segment by global index.
        Dynamically loads the song containing this segment.
        """
        # Find which song this segment belongs to
        song_idx = self.segment_to_song_idx[global_idx]
        
        # Load song data (only this song is in memory)
        song_data = self.musdb_sigsep[song_idx]
        song_segments = self.MusDB_segments(song_data)
        
        # Find local index within this song
        song_start_idx = sum(self.song_segment_counts[:song_idx])
        local_idx = global_idx - song_start_idx
        
        # Return the specific segment
        return song_segments[local_idx]


def get_dataloaders_ss(config):
    """
    Create dataloaders for source separation models using MusDB.
    
    Uses lazy-loading to avoid RAM overflow.
    Properly loads data using: musdb.DB → MusDBSigSepStems → MusDB_segments (on-demand)

    Args:
        config (DictConfig): hydra experiment config containing:
            - n_sources: 2 or 4
            - subsets: ['train', 'test'] or similar
            - dataloader: batch_size, num_workers, etc.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for each partition
    """
    import musdb
    
    dataloaders = {}
    n_sources = config.model.n_sources
    
    # Import the correct dataset classes based on n_sources
    if n_sources == 2:
        from src.datasets.musdb_sigsep_2sources import MusDBSigSepStems
    elif n_sources == 4:
        from src.datasets.musdb_sigsep_4sources import MusDBSigSep as MusDBSigSepStems
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
            
            # Use lazy-loading dataset (doesn't preload all songs into memory)
            lazy_dataset = LazyMusDBSegments(musdb_sigsep, n_sources=n_sources)
            
            if len(lazy_dataset) == 0:
                print(f"Warning: No segments found for subset '{subset}', skipping...")
                continue
            
            assert config.dataloader.batch_size <= len(lazy_dataset), (
                f"The batch size ({config.dataloader.batch_size}) cannot "
                f"be larger than the dataset length ({len(lazy_dataset)})"
            )
            
            # Create dataloader
            partition_dataloader = instantiate(
                config.dataloader,
                dataset=lazy_dataset,
                collate_fn=collate_fn_ss,
                drop_last=(subset == "train"),
                shuffle=(subset == "train"),
                worker_init_fn=set_worker_seed,
            )
            dataloaders[subset] = partition_dataloader
            
            print(f"Created dataloader for '{subset}' with {len(lazy_dataset)} segments from {len(musdb_sigsep)} songs")
        
        except Exception as e:
            print(f"Error loading subset '{subset}': {e}")
            import traceback
            traceback.print_exc()
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