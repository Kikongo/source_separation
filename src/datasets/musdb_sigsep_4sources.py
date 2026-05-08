import os
import zipfile
import musdb
from src.utils.io_utils import ROOT_PATH
import subprocess
from tqdm import tqdm
from pathlib import Path
import torch
import torchaudio
from src.datasets.musdb_utils import get_number_of_possible_segments
from src.datasets.musdb_utils import divide_into_segments
from torch.utils.data import Dataset

class MusDBSigSep(Dataset):
    def __init__(self, index):
        self.index = index

    def __len__(self):
        return len(self.index)


    def __getitem__(self, ind):
        item = self.index[ind]
        instance_data = {}

        source_audios = []
        source_spectrograms = []
        spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=512, win_length=2048)
        for key in item.targets:
            if key != 'accompaniment' and key != 'linear_mixture':
                #Get audio waveform and segments
                source_audio = item.targets[key].audio[:, 0] # sr = 44100
                source_audio = torch.tensor(source_audio, dtype=torch.float32).unsqueeze(0) # [1, n_samples]
                source_audios.append(source_audio)

                source_spectrogram_segments = []
                source_audio_segments = divide_into_segments(source_audio, sample_rate=44100)
                for source_audio_segment in source_audio_segments:
                    source_spectrogram_segment = spectrogram_transform(source_audio_segment)
                    source_spectrogram_segments.append(source_spectrogram_segment)

                source_segments_spectrogram = torch.stack(source_spectrogram_segments, dim=0) #[n_segm, freq, time]
                source_spectrograms.append(source_segments_spectrogram)

        stacked_source_spectrograms = torch.stack(source_spectrograms, dim=0) # [n_sources, n_segm, freg, time]
        instance_data["sources_segments_spectrograms"] = stacked_source_spectrograms.transpose(0,1)

        # Get mixture
        mixture_audio = item.audio[:, 0] # sr = 44100
        mixture_audio = torch.tensor(mixture_audio, dtype=torch.float32).unsqueeze(0) # [1, n_samples]
        mixture_spectrogram_segments = []
        mixture_audio_segments = divide_into_segments(mixture_audio, sample_rate=44100)
        for mixture_audio_segment in mixture_audio_segments:
            mixture_spectrogram_segment = spectrogram_transform(mixture_audio_segment)
            mixture_spectrogram_segments.append(mixture_spectrogram_segment)
        instance_data['mixture_segments_spectrograms'] = torch.stack(mixture_spectrogram_segments, dim=0) #[n_segm, freq, time]
       
        return instance_data