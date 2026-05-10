# Source Separation Training Guide

This guide explains how to use the source separation trainer to train models like in `model_2sources.ipynb` and `model_4sources.ipynb`.

## Quick Start

### Training a 2-Source Separation Model

```bash
python ss_train.py --config-name=ss_2sources
```

### Training a 4-Source Separation Model

```bash
python ss_train.py --config-name=ss_4sources
```

## Data Loading

The trainer automatically loads data using the MusDB structure on disk:
- `data/datasets/musdb18/train/` - Training songs
- `data/datasets/musdb18/test/` - Test songs

The data loading pipeline:
1. Load `musdb.DB` for the specified subset (train or test)
2. Wrap with `MusDBSigSepStems` to compute spectrograms for all sources
3. For each song, create `MusDB_segments` to split into fixed-size segments
4. Combine all song segments into a single dataset for batching

This matches exactly how you use the data in `model_2sources.ipynb`:
```python
mus_train = musdb.DB(subsets='train', download=False)
musdb_dataset = MusDBSigSepStems(mus_train)
for n in range(len(musdb_dataset)):
    song = MusDB_segments(musdb_dataset[n])
    dataloader = DataLoader(song, batch_size=batch_size)
```

## Configuration

The configuration files are located in `src/configs/`:

- `ss_2sources.yaml` - Configuration for 2-source separation (vocals + accompaniment)
- `ss_4sources.yaml` - Configuration for 4-source separation (vocals, drums, bass, other)

### Key Configuration Parameters

#### Model
- `n_sources`: Number of sources to separate (2 or 4)

#### Training
- `n_epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `learning_rate` (lr): Learning rate for optimizer
- `weight_decay`: L2 regularization parameter
- `monitor`: Metric to monitor for early stopping (e.g., "min val_loss")
- `early_stop`: Number of epochs without improvement before stopping

#### Data
- `subsets`: Which MusDB subsets to use (e.g., ["train", "test"])

#### Loss Function
The `SourceSeparationLoss` uses weighted L1 loss:
- `alpha`: Weight for each source
  - For 2 sources: Single value (e.g., 0.707) - first source gets this weight, second gets 1-alpha
  - For 4 sources: List of 4 values (e.g., [0.25, 0.25, 0.25, 0.25])

#### Metrics
Two main metrics are available:
- `SourceSeparationL1Metric`: L1 error between predicted and true sources
- `SourceSeparationMetric`: Correlation between predicted and true sources

## Training Process

The trainer performs the following during each epoch:

1. **Forward Pass**: Input mixture spectrogram → Model → Predicted source spectrograms
2. **Loss Computation**: Weighted L1 loss across all sources
3. **Backward Pass**: Compute gradients
4. **Optimization**: Update model parameters with AdamW optimizer
5. **Evaluation**: Evaluate on test set
6. **Logging**: Log metrics and spectrograms to wandb

## Model Checkpoints

Checkpoints are saved in:
```
outputs/<date>/<time>/checkpoints/
```

Checkpoint includes:
- Model state dict
- Optimizer state dict
- Learning rate scheduler state
- Configuration
- Best monitored metric value

## Resuming Training

To resume training from a checkpoint:

```yaml
trainer:
  resume_from: "checkpoint-epoch10.pth"
```

To use a pretrained model:

```yaml
trainer:
  from_pretrained: "/path/to/pretrained_model.pth"
```

## Custom Loss Functions

### SourceSeparationLoss
Basic weighted L1 loss for each source:

```python
from src.loss.source_separation_loss import SourceSeparationLoss

loss = SourceSeparationLoss(n_sources=2, alpha=0.707)
```

### SourceSeparationLossWithConsistency
Adds consistency regularization to ensure predicted sources sum to the mixture:

```python
from src.loss.source_separation_loss import SourceSeparationLossWithConsistency

loss = SourceSeparationLossWithConsistency(
    n_sources=2,
    alpha=0.707,
    consistency_weight=0.1
)
```

## Monitoring Training

Training progress is logged to Weights & Biases (wandb). Key metrics include:

- `loss`: Total loss
- `loss_source_0`, `loss_source_1`, etc.: Per-source losses
- `train_l1_error`: Training L1 error
- `val_l1_error`: Test L1 error
- `val_correlation`: Test correlation between predictions and ground truth
- `learning_rate`: Current learning rate

## Evaluation

After training, evaluate on a test song using:

```python
import torch
from torch.utils.data import DataLoader
import musdb
from src.datasets.musdb_sigsep_2sources import MusDBSigSepStems
from src.datasets.musdb_segments_2sources import MusDB_segments
from src.model.mss_model_final import StandardSeparationUNet

# Load model
model = StandardSeparationUNet(n_sources=2)
model.load_state_dict(torch.load('model_best.pth')['state_dict'])
model.eval()

# Load data
mus_test = musdb.DB(subsets='test', download=False)
musdb_dataset = MusDBSigSepStems(mus_test)
dataset = MusDB_segments(musdb_dataset[0])
dataloader = DataLoader(dataset, batch_size=8)

# Inference
with torch.no_grad():
    for mixture, sources in dataloader:
        mixture = mixture.unsqueeze(1).to('cuda')
        predictions = model(mixture)
        # Process predictions...
```

## Troubleshooting

### Out of Memory (OOM) Error
- Reduce `batch_size` in configuration
- Enable `skip_oom` in trainer config to skip problematic batches
- Use gradient accumulation

### Slow Training
- Reduce number of workers in dataloader
- Use mixed precision training
- Reduce validation frequency

### Poor Performance
- Increase learning rate or adjust learning rate schedule
- Try different alpha weights for loss function
- Increase number of training epochs
- Verify dataset is correctly loaded

### Data Loading Errors
- Ensure `data/datasets/musdb18/` directory exists with train/test subdirectories
- Verify you have MusDB18 dataset downloaded (see MusDB documentation)
- Check MUSDB_PATH environment variable if needed

## Differences from Notebook Implementation

The trainer implementation provides several improvements over the notebook code:

1. **Scalability**: Handles arbitrary number of sources
2. **Configurability**: Fully configurable via YAML files
3. **Monitoring**: Automatic metric tracking and early stopping
4. **Checkpointing**: Full experiment reproducibility
5. **Logging**: Integration with wandb for experiment tracking
6. **Evaluation Metrics**: Built-in correlation and L1 metrics
7. **Multi-partition Support**: Separate train/test partitions
8. **Learning Rate Scheduling**: MultiStepLR scheduler with milestone support

## Output Structure

```
outputs/
└── <YYYY-MM-DD>/
    └── <HH-MM-SS>/
        ├── config.yaml              # Experiment configuration
        ├── checkpoints/
        │   ├── checkpoint-epoch1.pth
        │   ├── checkpoint-epoch5.pth
        │   └── model_best.pth
        └── logs/
            └── ... (wandb logs)
```

## Environment Variables

The trainer respects the following environment variables:

- `MUSDB_PATH`: Path to MusDB database (defaults to checking standard locations)
- `CUDA_VISIBLE_DEVICES`: GPU device to use
