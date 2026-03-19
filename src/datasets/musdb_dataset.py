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

class MusDB(Dataset):
    def __init__(self, data_path=None, subsets=None):
        self.data_path = ROOT_PATH / 'data' / 'datasets' / 'musdb' / 'musdb18'
        self.wav_path = ROOT_PATH / 'data' / 'datasets' / 'musdb' / 'musdb18_wav'
        self.target_sr = 16000
        # if not self.wav_path.exists():
        #     self.wav_path.mkdir(exist_ok=True, parents=True)

        #     subprocess.run(
        #                         [
        #             "musdbconvert",
        #             self.data_path,
        #             self.wav_path,
        #         ]
        #     )

        self.subsets = subsets
        if subsets is None:
            subsets = 'train'

        self.index = self.__create_index(subsets)

    def __len__(self):
        return len(self.index)

    def __create_index(self, subsets):
        index = []
        wav_dirs = set()
        split_dir = self.wav_path / subsets
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".wav") for f in filenames]):
                wav_dirs.add(dirpath)
        for wav_dir in tqdm(
            list(wav_dirs), desc=f"Preparing musdb folders: {subsets}"
        ):
            wav_dir = Path(wav_dir)
            accompaniment_path = wav_dir / 'accompaniment.wav'
            bass_path = wav_dir / 'bass.wav'
            drums_path = wav_dir / 'drums.wav'
            other_path = wav_dir / 'other.wav'
            vocals_path = wav_dir / 'vocals.wav'
            mixture_path = wav_dir / 'mixture.wav'
            index.append({
                'accompaniment': str(accompaniment_path.absolute().resolve()),
                'bass': str(bass_path.absolute().resolve()),
                'drums': str(drums_path.absolute().resolve()),
                'other': str(other_path.absolute().resolve()),
                'vocals': str(vocals_path.absolute().resolve()),
                'mixture': str(mixture_path.absolute().resolve())
            })
        return index

    def __getitem__(self, ind):
        item = self.index[ind]
        instance_data = {}

        source_audios = []
        source_spectrograms = []
        for key in item:
            if key == 'mixture' or key == 'accompaniment':
                #Get audio waveform and segments
                instance_data[f"{key}_audio"] = self.load_audio(item[key])

                spectrogram_segments = []
                audio_segments = divide_into_segments(instance_data[f"{key}_audio"])

                instance_data[f"{key}_segments_audio"] = audio_segments
                
                for audio_segment in audio_segments:
                    spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000)
                    spectrogram_segment = spectrogram_transform(audio_segment)
                    spectrogram_segments.append(spectrogram_segment)
                
                instance_data[f"{key}_segments_spectrograms"] = torch.stack(spectrogram_segments, dim=0) #[n_segm, freq, time]
            else:
                source_audio = self.load_audio(item[key])
                source_audios.append(source_audio)

                source_spectrogram_segments = []
                source_audio_segments = divide_into_segments(source_audio)
                for source_audio_segment in source_audio_segments:
                    source_spectrogram_segment = spectrogram_transform(source_audio_segment)
                    source_spectrogram_segments.append(source_spectrogram_segment)

                source_segments_spectrogram = torch.stack(source_spectrogram_segments, dim=0) #[n_segm, freq, time]
                source_spectrograms.append(source_segments_spectrogram)

        stacked_source_spectrograms = torch.stack(source_spectrograms, dim=0) # [n_sources, n_segm, freg, time]
        instance_data["source_audios"] = source_audios
        instance_data["source_segments_spectrograms"] = stacked_source_spectrograms.transpose(0,1)
        instance_data['path'] = item['mixture'] # path to the file, maybe will need this
        return instance_data
    
    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path) # sr = 44100
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor