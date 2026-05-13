"""
Inference script for source separation models.
Demonstrates how to use trained models for full song separation and reconstruction.
"""

import os
import torch
import numpy as np
import librosa
from pathlib import Path
from torch.utils.data import DataLoader

# Set MUSDB path
os.environ['MUSDB_PATH'] = '/source_separation/data/'


def load_model(checkpoint_path, n_sources=2, device='cuda'):
    """
    Load a trained source separation model.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        n_sources (int): Number of sources
        device (str): Device to load model on
    
    Returns:
        model: Loaded model on the specified device
    """
    from src.model.mss_model_final import StandardSeparationUNet
    
    model = StandardSeparationUNet(n_sources=n_sources).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    return model


def separate_full_song(model, dataset, batch_size=8, device='cuda'):
    """
    Separate a full song into its sources using the trained model.
    
    Args:
        model: Trained separation model
        dataset: Dataset containing the song (e.g., MusDB_segments)
        batch_size (int): Batch size for inference
        device (str): Device to use for inference
    
    Returns:
        dict: Dictionary with keys 'sources_<i>' containing separated source spectrograms
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_sources = {i: [] for i in range(model.module.n_sources if hasattr(model, 'module') else 2)}
    
    model.eval()
    with torch.no_grad():
        for mixture, _ in dataloader:
            mixture = mixture.unsqueeze(1).to(device)  # [B, 1, F, T]
            predictions = model(mixture)  # [B, n_sources, F, T]
            
            # Collect predictions for each source
            for source_idx in range(predictions.shape[1]):
                all_sources[source_idx].append(predictions[:, source_idx].cpu().numpy())
    
    # Stack segments horizontally
    result = {}
    for source_idx in range(len(all_sources)):
        stacked = np.hstack(all_sources[source_idx])
        result[f'source_{source_idx}'] = stacked
    
    return result


def reconstruct_audio(
    source_spectrogram,
    original_audio,
    n_fft=2048,
    hop_length=512,
    win_length=2048,
    sr=44100
):
    """
    Reconstruct audio from predicted source spectrogram using original phase.
    
    Args:
        source_spectrogram (np.ndarray): Predicted source spectrogram [F, T]
        original_audio (np.ndarray): Original audio for phase extraction [samples]
        n_fft (int): FFT size
        hop_length (int): Hop length
        win_length (int): Window length
        sr (int): Sample rate
    
    Returns:
        np.ndarray: Reconstructed audio waveform
    """
    # Compute STFT of original audio
    stft_original = librosa.stft(
        original_audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
    
    # Extract phase
    _, phase = librosa.magphase(stft_original)
    
    # Apply phase to predicted magnitude
    complex_spec = source_spectrogram * phase
    
    # Inverse STFT
    audio = librosa.istft(
        complex_spec,
        hop_length=hop_length,
        win_length=win_length
    )
    
    return audio


def main():
    """
    Example: Load model, separate a song, and reconstruct audio.
    """
    # Configuration
    checkpoint_path = 'model_best.pth'
    n_sources = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    song_index = 5  # Which song in MusDB to process
    
    # Load model
    model = load_model(checkpoint_path, n_sources=n_sources, device=device)
    
    # Load dataset
    import musdb
    from src.datasets.musdb_sigsep_2sources import MusDBSigSepStems
    from src.datasets.musdb_segments_2sources import MusDB_segments
    
    mus_train = musdb.DB(subsets='train', download=False)
    musdb_dataset = MusDBSigSepStems(mus_train)
    
    # Get dataset for a specific song
    dataset = MusDB_segments(musdb_dataset[song_index])
    
    print(f"Processing song {song_index} with {len(dataset)} segments...")
    
    # Separate song
    separated_sources = separate_full_song(model, dataset, batch_size=8, device=device)
    
    print(f"Separated sources: {list(separated_sources.keys())}")
    print(f"Spectrogram shapes: {[(k, v.shape) for k, v in separated_sources.items()]}")
    
    # Reconstruct audio
    original_audio = mus_train[song_index].audio[:, 0]  # Get mono mix
    
    reconstructed_audio = {}
    for source_name, spec in separated_sources.items():
        # Limit spectrogram to original size
        spec = spec[:, :librosa.stft(original_audio, n_fft=2048, hop_length=512).shape[1]]
        
        audio = reconstruct_audio(spec, original_audio)
        reconstructed_audio[source_name] = audio
        
        print(f"Reconstructed {source_name}: {audio.shape}")
        
        # Optional: Save audio
        # import soundfile
        # soundfile.write(f'{source_name}.wav', audio, sr=44100)
    
    return reconstructed_audio


if __name__ == "__main__":
    audio = main()
    print("Inference completed!")
