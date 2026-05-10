import numpy as np
import torch
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

    # def zero_pad_segments():
    #     '''
    #     this function will add zeros to the end of the segments
    #     if `zero_pad_factor` = 0, number of zeros added to the end is `int(self.sample_rate*segment_length_in_seconds) - len(segments[-1])`
    #     which makes all segments to have an equal length. otherwise, the number of zeros to add is `int(self.sample_rate*zero_pad_factor*segment_length_in_seconds) - len(segments[-1])
    #     have a look at https://www.bitweenie.com/listings/fft-zero-padding/ for zero padding reasons
    #     '''
    #     number_of_needed_samples_for_the_last_segment = int(sample_rate*segment_length_in_seconds)-len(segments[-1])
    #     silence_to_last_segment = librosa.tone(0.0, sr=sample_rate, length=number_of_needed_samples_for_the_last_segment) #create a silence tone with the length of the needed number of samples to make the last segment have same length as the other segments
    #     segments[-1] = np.concatenate((segments[-1], silence_to_last_segment)) #add the silence to the last segment
    #     #adds the paddings to the segment arrays based on zero_pad_factor value
    #     for i in range(zero_pad_factor):
    #         segments.append(librosa.tone(0.0, sr=sample_rate, length=sample_rate*segment_length_in_seconds))
    #     return segments

    # segments = zero_pad_segments()
    pad_size = int(sample_rate * segment_length_in_seconds) - len(segments[-1])
    padded_tensor = torch.nn.functional.pad(segments[-1], (0, pad_size), "constant", 0)
    segments[-1] = padded_tensor
    return torch.stack(segments, dim=0)