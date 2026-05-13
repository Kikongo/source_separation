import torch
import numpy as np
import torchaudio

def stft(waveform):
    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=512, win_length=2048, power=None)
    complex_spec = spectrogram_transform(waveform)

    # Extract Magnitude (1, freq, time)
    magnitude = torch.abs(complex_spec)

    # Extract Phase (1, freq, time)
    phase = torch.angle(complex_spec)

    return magnitude, phase

def istft(spectrogram, phase, waveform_length):
    spectrogram = torch.sqrt(spectrogram)
    complex_spectrogram = torch.multiply(spectrogram, phase)
    complex_spectrogram = complex_spectrogram.to(torch.cdouble)
    waveform_length = int(waveform_length)

    inverse_spectrogram_transform = torchaudio.transforms.InverseSpectrogram(
        n_fft=2048,
        hop_length=512,
        win_length=2048
        # window_fn=torch.hann_window # Default is hann_window, explicitly setting if needed for consistency
    )

    # Pass the inferred length
    waveform = inverse_spectrogram_transform(complex_spectrogram, length=waveform_length)
    return waveform

def get_full_audios(model, segments_spectrograms, num_sources, orig_spectrogram_length):
    full_output_specs = {}

    for i in range(num_sources):
        full_output_specs[i] = []
    
    for segment_spectrogram in segments_spectrograms:
        segment_spectrogram = segment_spectrogram.cuda()
        output_specs = model(segment_spectrogram.unsqueeze(0)) #[Batch, N_sources, freq, time]
        for i in num_sources:
            full_output_specs[i].append(output_specs[0][i])
    for i in num_sources:
        full_output_specs[i] = torch.hstack(full_output_specs[i][:, :orig_spectrogram_length])

def get_segments_spectrograms(mixture_audio):
    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=512, win_length=2048)

    # Get mixture
    #mixture_audio = torch.tensor(mixture_audio, dtype=torch.float32).unsqueeze(0) # [1, n_samples]
    mixture_spectrogram_segments = []
    mixture_audio_segments = divide_into_segments(mixture_audio, sample_rate=44100)
    for mixture_audio_segment in mixture_audio_segments:
        mixture_spectrogram_segment = spectrogram_transform(mixture_audio_segment)
        mixture_spectrogram_segments.append(mixture_spectrogram_segment)

    mixture_spectrogram_segments = torch.stack(mixture_spectrogram_segments, dim=0) #[n_segm, freq, time]

    return mixture_spectrogram_segments


def get_number_of_possible_segments(wav_file, segment_length_in_seconds, sample_rate):
    return int(np.ceil(len(wav_file) / (sample_rate*segment_length_in_seconds)))

def divide_into_segments(wav_file, sample_rate = 16000,segment_length_in_seconds:float=2.0, zero_pad_factor:int=0):
    wav_file = wav_file[0]
    wav_file_duration = len(wav_file) / sample_rate
    if segment_length_in_seconds <= 0.0 or segment_length_in_seconds > wav_file_duration:

        return np.array([wav_file])

        #if segment_length_in_seconds is 0.0 or less, or greater than the duration of audio file, no segmentation is performed
        #first the self.__wav_file will be converted to a list that contains it and then a numpy array of that list is returned
        #the reason is that we want to treat self.__wav_file as one segment so that in other methods, we still have a segments list that we can loop through

    number_of_segments_in_file = get_number_of_possible_segments(wav_file, segment_length_in_seconds, sample_rate)
    #the number of seconds in a wave file is 1/sample_rate. if we divide that by the length of chunks we want,
    #we get the number of chunks with that length in our audio file

    segments = []
    for i in range(0, number_of_segments_in_file):
        segments.append(wav_file[int(i*sample_rate*segment_length_in_seconds): int((i+1)*sample_rate*segment_length_in_seconds)])

    # segments = zero_pad_segments()
    pad_size = int(sample_rate * segment_length_in_seconds) - len(segments[-1])
    padded_tensor = torch.nn.functional.pad(segments[-1], (0, pad_size), "constant", 0)
    segments[-1] = padded_tensor

    return torch.stack(segments, dim=0)