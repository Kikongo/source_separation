import torch
import torch.nn as nn

# Обычная U-Net для прямого разделения
class StandardSeparationUNet(nn.Module):
    def __init__(self, n_sources=4):
        super().__init__()
        self.n_sources = n_sources

        # Batch norm
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(0.4)

        # Энкодер
        self.enc1_1 = nn.Conv2d(1, 32, 3, padding=1)  # [batch, 64, freq, time]
        self.enc1_2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.enc1_3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.enc2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.enc2_2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.enc2_3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)

        self.enc3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc3_2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.enc3_3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)

        self.enc4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.enc4_2 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.enc4_3 = nn.Conv2d(256, 256, 3, stride=2, padding=1)

        # Боттлнек
        self.bottleneck = nn.Conv2d(256, 512, 3, padding=1)
        self.batch_norm_center = nn.BatchNorm2d(512)

        # Декодер
        self.dec1_1 = nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1, output_padding=0)
        self.dec1_2 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=(0, 1))
        self.dec1_3 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=(1, 0))

        self.dec2_1 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1, output_padding=0)
        self.dec2_2 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=(1, 1))
        self.dec2_3 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=(1, 1))

        self.dec3_1 = nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1, output_padding=0)
        self.dec3_2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=(1, 1))
        self.dec3_3 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=(1, 0))        

        self.dec4_1 = nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1, output_padding=0)
        self.dec4_2 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=(1, 0))
        self.dec4_3 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=(1, 1))

        # Выходной слой - сразу все источники
        self.output = nn.Conv2d(32, n_sources, kernel_size=(1,1))  # [batch, n_sources, freq, time]

    def forward(self, mixture_spectrogram):
        # mixture_spectrogram: [batch, 1, freq, time] - спектрограмма смеси

        # Прямой проход через U-Net
        e1_1 = torch.relu(self.batch_norm1(self.enc1_1(mixture_spectrogram)))
        e1_2 = torch.relu(self.batch_norm1(self.enc1_2(e1_1)))
        e1_3 = torch.relu(self.batch_norm1(self.enc1_3(e1_2)))

        print(e1_3.shape)
        e2_1 = torch.relu(self.batch_norm2(self.enc2_1(e1_3)))
        e2_2 = torch.relu(self.batch_norm2(self.enc2_2(e2_1)))
        e2_3 = torch.relu(self.batch_norm2(self.enc2_3(e2_2)))

        print(e2_3.shape)
        e3_1 = torch.relu(self.batch_norm3(self.enc3_1(e2_3)))
        e3_2 = torch.relu(self.batch_norm3(self.enc3_2(e3_1)))
        e3_3 = torch.relu(self.batch_norm3(self.enc3_3(e3_2)))

        print(e3_3.shape)

        e4_1 = torch.relu(self.batch_norm4(self.enc4_1(e3_3)))
        e4_2 = torch.relu(self.batch_norm4(self.enc4_2(e4_1)))
        e4_3 = torch.relu(self.batch_norm4(self.enc4_3(e4_2)))

        print(e4_3.shape)

        b = torch.relu(self.batch_norm_center(self.bottleneck(e4_3)))
        print(f"bottlenec {b.shape}")

        d1_1 = self.dropout(torch.relu(self.batch_norm4(self.dec1_1(b))))
        d1_2 = torch.relu(self.batch_norm4(self.dec1_2(d1_1 + e4_3)))
        d1_3 = torch.relu(self.batch_norm4(self.dec1_3(d1_2)))

        print(f"dec1 {d1_3.shape}")
        
        d2_1 = self.dropout(torch.relu(self.batch_norm3(self.dec2_1(d1_3))))  # Skip connection
        d2_2 = torch.relu(self.batch_norm3(self.dec2_2(d2_1 + e3_3)))  # Skip connection
        d2_3 = torch.relu(self.batch_norm3(self.dec2_3(d2_2)))  # Skip connection
        
        print(f"dec2 {d2_3.shape}")
        d3_1 = self.dropout(torch.relu(self.batch_norm2(self.dec3_1(d2_3))))  # Skip connection
        d3_2 = torch.relu(self.batch_norm2(self.dec3_2(d3_1 + e2_3)))  # Skip connection
        d3_3 = torch.relu(self.batch_norm2(self.dec3_3(d3_2)))  # Skip connection

        print(f"dec3 {d3_3.shape}")

        d4_1 = self.dropout(torch.relu(self.batch_norm1(self.dec4_1(d3_3))))  # Skip connection
        d4_2 = torch.relu(self.batch_norm1(self.dec4_2(d4_1 + e1_3)))  # Skip connection
        d4_3 = torch.relu(self.batch_norm1(self.dec4_3(d4_2)))  # Skip connection

        print(f"dec4 {d4_3.shape}")

        # Выход: маски или спектрограммы для всех источников
        output = torch.sigmoid(self.output(d4_3))
        print(f"output {output.shape}")
        return output  # [batch, n_sources, freq, time]
