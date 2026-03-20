import torch
import torch.nn as nn

# Обычная U-Net для прямого разделения
class StandardSeparationUNet(nn.Module):
    def __init__(self, n_sources=4):
        super().__init__()
        self.n_sources = n_sources

        # Энкодер
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)  # [batch, 64, freq, time]
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        # Боттлнек
        self.bottleneck = nn.Conv2d(256, 512, 3, padding=1)

        # Декодер
        self.dec1 = nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1, output_padding=0)
        self.dec2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=(1, 0))
        self.dec3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=(1, 1))

        # Выходной слой - сразу все источники
        self.output = nn.Conv2d(64, n_sources, kernel_size=(1,1))  # [batch, n_sources, freq, time]

    def forward(self, mixture_spectrogram):
        # mixture_spectrogram: [batch, 1, freq, time] - спектрограмма смеси

        # Прямой проход через U-Net
        e1 = torch.relu(self.enc1(mixture_spectrogram))
        print(e1.shape)
        e2 = torch.relu(self.enc2(e1))
        print(e2.shape)
        e3 = torch.relu(self.enc3(e2))
        print(e3.shape)

        b = torch.relu(self.bottleneck(e3))
        print(f"bottlenec {b.shape}")

        d1 = torch.relu(self.dec1(b))
        print(f"dec1 {d1.shape}")
        d2 = torch.relu(self.dec2(d1 + e3))  # Skip connection
        print(f"dec2 {d2.shape}")
        d3 = torch.relu(self.dec3(d2 + e2))  # Skip connection
        print(f"dec3 {d3.shape}")

        # Выход: маски или спектрограммы для всех источников
        output = torch.sigmoid(self.output(d3))
        print(f"output {output.shape}")
        return output  # [batch, n_sources, freq, time]
