from torch import nn
import torch

class DeepSpeech2(nn.Module):
    """
    DeepSpeech2 model
    """

    def __init__(
        self,
        n_feats,
        n_tokens,
        rnn_hidden=800,
        rnn_layers=7,
        rnn_type="lstm",
        bidirectional=True,
        conv_channels=32,
        conv_kernel=(41, 11),
        conv_stride=(2, 2),
        dilation = (1, 1),
        conv_padding=(20, 10),
        conv_kernel2 = (22, 11),
        conv_stride2 = (2, 1),
        conv_padding2 = (10, 5),
    ):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            rnn_hidden (int): number of hidden features in RNN layers.
            rnn_layers (int): number of RNN layers.
            rnn_type (str): type of RNN layer (lstm, gru, rnn_tanh, rnn_relu).
            bidirectional (bool): if True, RNN layers will be bidirectional.
            conv_channels (int): number of convolution channels.
            conv_kernel (tuple): convolution kernel size.
            conv_stride (tuple): convolution stride size.
            conv_padding (tuple): convolution padding size.
        """
        super().__init__()

        self.conv_padding = conv_padding
        self.conv_padding2 = conv_padding2
        self.conv_kernel = conv_kernel
        self.conv_kernel2 = conv_kernel2
        self.conv_stride = conv_stride
        self.conv_stride2 = conv_stride2
        self.dilation = dilation

        self.subsampling = nn.Conv2d(
            n_feats,
            conv_channels,
            conv_kernel,
            conv_stride,
            conv_padding,
            bias=False,
        )

        self.batch_norm = nn.BatchNorm2d(conv_channels)
        self.tanh = nn.Hardtanh()

        self.subsampling2 = nn.Conv2d(
            conv_channels,
            conv_channels,
            conv_kernel2,
            conv_stride2,
            conv_padding2,
            bias=False,
        )

        rnn_input_size = conv_channels * 32

        self.rnns = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=bidirectional,
            dropout=0.1,
            bias=True,
        )

        rnn_output_size = rnn_hidden * 2 if bidirectional else rnn_hidden

        self.fc = nn.Linear(in_features=rnn_output_size, out_features=n_tokens)
    
    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram. (B, C, F, T) F=128
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        #x = spectrogram.transpose(1, 2).unsqueeze(1)  # (B, 1, T, F)
        x = spectrogram # (B, 1, F, T)
        x = self.subsampling(x)  # (B, C, F', T')

        x = self.batch_norm(x)
        x = self.tanh(x)
        x = self.subsampling2(x)
        x = self.batch_norm(x)
        x = self.tanh(x)

        batch_size, channels, dimension, seq_length = x.size()
        print(x.shape)
        #x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, dimension, channels * seq_length)
        x = x.view(batch_size, channels * dimension, seq_length)
        print(x.shape)
        x = x.transpose(2, 1)
        print(x.shape)

        out = ((spectrogram_length + 2 * self.conv_padding[1] - self.dilation[1] * (self.conv_kernel[1] - 1) - 1) // self.conv_stride[1] + 1)
        out = ((out + 2 * self.conv_padding2[1] - self.dilation[1] * (self.conv_kernel2[1] - 1) - 1) // self.conv_stride2[1] + 1)
        seq_batch_len = torch.full((batch_size,), out)
        x = nn.utils.rnn.pack_padded_sequence(x, seq_batch_len, batch_first=True)
        x, _ = self.rnns(x)
        x, output_lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = self.fc(x)
        print(f"x output {x.shape}")
        log_probs = nn.functional.log_softmax(x, dim=-1)

        return {"log_probs": log_probs, "log_probs_length": output_lengths}