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

class MusDBSigSepStems(Dataset):
    def __init__(self, index):
        self.index = index

    def __len__(self):
        return len(self.index)


    def __getitem__(self, ind):
        item = self.index[ind]
        instance_data = {}

        spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=512, win_length=2048)
        # for key in item.targets:
        #     #Get audio waveform and segments
        #     instance_data[f"{key}_audio"] = item.targets[key].audio[:, 0] # sr = 44100
        #     instance_data[f"{key}_audio"] = torch.tensor(instance_data[f"{key}_audio"], dtype=torch.float32).unsqueeze(0) # [1, n_samples]
            
        #     spectrogram_segments = []
        #     audio_segments = divide_into_segments(instance_data[f"{key}_audio"], sample_rate=44100)

        #     instance_data[f"{key}_segments_audio"] = audio_segments
            
        #     for audio_segment in audio_segments:
        #         spectrogram_segment = spectrogram_transform(audio_segment)
        #         spectrogram_segments.append(spectrogram_segment)
            
        #     instance_data[f"{key}_segments_spectrograms"] = torch.stack(spectrogram_segments, dim=0) #[n_segm, freq, time]

        vocals_audio = item.targets["vocals"].audio[:, 0] # sr = 44100
        vocals_audio = torch.tensor(vocals_audio, dtype=torch.float32).unsqueeze(0) # [1, n_samples]
        
        vocal_spectrogram_segments = []
        audio_segments = divide_into_segments(vocals_audio, sample_rate=44100)
        
        for audio_segment in audio_segments:
            spectrogram_segment = spectrogram_transform(audio_segment)
            vocal_spectrogram_segments.append(spectrogram_segment)

        vocal_spectrogram_segments = torch.stack(vocal_spectrogram_segments, dim=0)

        acc_audio = item.targets["accompaniment"].audio[:, 0] # sr = 44100
        acc_audio = torch.tensor(acc_audio, dtype=torch.float32).unsqueeze(0) # [1, n_samples]
        
        acc_spectrogram_segments = []
        audio_segments = divide_into_segments(acc_audio, sample_rate=44100)
        
        for audio_segment in audio_segments:
            spectrogram_segment = spectrogram_transform(audio_segment)
            acc_spectrogram_segments.append(spectrogram_segment)

        acc_spectrogram_segments = torch.stack(acc_spectrogram_segments, dim=0)
        
        stacked_source_spectrograms = torch.stack((vocal_spectrogram_segments, acc_spectrogram_segments), dim=0)
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