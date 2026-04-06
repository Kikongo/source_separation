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
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)

        # Энкодер
        self.enc1_1 = nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False)  # [batch, 64, freq, time]
        self.enc1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)

        self.enc2_1 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.enc2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)

        self.enc3_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.enc3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)

        self.enc4_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False)
        self.enc4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)

        # Боттлнек
        self.bottleneck = nn.Conv2d(256, 512, 3, padding=1, bias=False)
        self.bottleneck2 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.batch_norm_center = nn.BatchNorm2d(512)

        # Декодер
        self.dec1_1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=(1, 1))
        self.dec1_2 = nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, output_padding=0)
        self.dec1_3 = nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, output_padding=0)

        self.dec2_1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=(1, 1))
        self.dec2_2 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, output_padding=0)
        self.dec2_3 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, output_padding=0)

        self.dec3_1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=(1, 1))
        self.dec3_2 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, output_padding=0)
        self.dec3_3 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, output_padding=0)

        self.dec4_1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=(1, 1))
        self.dec4_2 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1, output_padding=0)
        self.dec4_3 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1, output_padding=0)

        # Выходной слой - сразу все источники
        self.output = nn.Conv2d(32, n_sources+1, kernel_size=(1,1))  # [batch, n_sources, freq, time]

    def forward(self, mixture_spectrogram):
        # mixture_spectrogram: [batch, 1, freq, time] - спектрограмма смеси
        
        #padding the height
        if mixture_spectrogram.shape[2] % 16 != 0:
            pad_to_add = 16 - mixture_spectrogram.shape[2] % 16
            #the following block tries to pad the data symmetrically and not only to one side
            pad_to_top = pad_to_add // 2
            pad_to_bottom = pad_to_add - pad_to_top
            mixture_spectrogram = nn.ZeroPad2d((0, 0, pad_to_top, pad_to_bottom))(mixture_spectrogram)

        #padding the width
        if mixture_spectrogram.shape[3] % 16 != 0:
            pad_to_add = 16 - mixture_spectrogram.shape[3] % 16
            #the following block tries to pad the data symmetrically and not only to one side
            pad_to_left = pad_to_add // 2
            pad_to_right = pad_to_add - pad_to_left
            mixture_spectrogram = nn.ZeroPad2d((pad_to_left, pad_to_right))(mixture_spectrogram)


        # Прямой проход через U-Net
        e1_1 = torch.relu(self.batch_norm1(self.enc1_1(mixture_spectrogram)))
        e1_2 = torch.relu(self.batch_norm1(self.enc1_2(e1_1)))
        max_pool_e1 = self.max_pool(e1_2)

        #print(f"Enc1_1 {e1_1.shape}")
        #print(f"MaxPoolE1 {max_pool_e1.shape}")

        e2_1 = torch.relu(self.batch_norm2(self.enc2_1(max_pool_e1)))
        e2_2 = torch.relu(self.batch_norm2(self.enc2_2(e2_1)))
        max_pool_e2 = self.max_pool(e2_2)

        #print(f"Enc2_1 {e2_1.shape}")
        #print(f"MaxPoolE2 {max_pool_e2.shape}")

        e3_1 = torch.relu(self.batch_norm3(self.enc3_1(max_pool_e2)))
        e3_2 = torch.relu(self.batch_norm3(self.enc3_2(e3_1)))
        max_pool_e3 = self.max_pool(e3_2)

        #print(f"Enc3_1 {e3_1.shape}")
        #print(f"MaxPoolE3 {max_pool_e3.shape}")

        e4_1 = torch.relu(self.batch_norm4(self.enc4_1(max_pool_e3)))
        e4_2 = torch.relu(self.batch_norm4(self.enc4_2(e4_1)))
        max_pool_e4 = self.max_pool(e4_2)

        #print(f"Enc4_1 {e4_1.shape}")
        #print(f"MaxPoolE4 {max_pool_e4.shape}")

        b = torch.relu(self.batch_norm_center(self.bottleneck(max_pool_e4)))
        #print(f"bottlenec {b.shape}")
        b = torch.relu(self.batch_norm_center(self.bottleneck2(b)))
        #print(f"bottlenec {b.shape}")

        d1_1 = self.dropout(torch.relu(self.batch_norm4(self.dec1_1(b))))
        #print(f"dec1 {d1_1.shape}")
        d1_2 = torch.relu(self.batch_norm4(self.dec1_2(d1_1 + e4_1)))
        #print(f"dec1_2 {d1_2.shape}")
        d1_3 = torch.relu(self.batch_norm4(self.dec1_3(d1_2)))

        d2_1 = self.dropout(torch.relu(self.batch_norm3(self.dec2_1(d1_3))))  # Skip connection
        d2_2 = torch.relu(self.batch_norm3(self.dec2_2(d2_1 + e3_1)))  # Skip connection
        d2_3 = torch.relu(self.batch_norm3(self.dec2_3(d2_2)))  # Skip connection
        #print(f"dec2 {d2_1.shape}")

        d3_1 = self.dropout(torch.relu(self.batch_norm2(self.dec3_1(d2_3))))  # Skip connection
        d3_2 = torch.relu(self.batch_norm2(self.dec3_2(d3_1 + e2_1)))  # Skip connection
        d3_3 = torch.relu(self.batch_norm2(self.dec3_3(d3_2)))  # Skip connection
        #print(f"dec3 {d3_1.shape}")

        d4_1 = self.dropout(torch.relu(self.batch_norm1(self.dec4_1(d3_3))))  # Skip connection
        d4_2 = torch.relu(self.batch_norm1(self.dec4_2(d4_1 + e1_1)))  # Skip connection
        d4_3 = torch.relu(self.batch_norm1(self.dec4_3(d4_2)))  # Skip connection
        #print(f"dec4 {d4_1.shape}")

        # Выход: маски или спектрограммы для всех источников
        output = torch.relu(self.output(d4_3))
        #print(f"output {output.shape}")
        
        # Crop spectrogram to original size
        output = output[:,:,pad_to_top:-pad_to_bottom, pad_to_left:-pad_to_right]

        return output  # [batch, n_sources, freq, time]
