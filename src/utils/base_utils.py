from IPython import display
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_waveform(waveform:torch.tensor, sample_rate=44100, title="Waveform", xlim=None, ylim=None):
    time_axis = np.arange(0, waveform.shape[1]) / sample_rate

    plt.figure(figsize=(12, 4))
    plt.title(title)
    plt.plot(time_axis, waveform[0])
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    display.display(display.Audio(waveform, rate=sample_rate, normalize=True))