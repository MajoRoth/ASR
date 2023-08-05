"""

    not working, some experiment i made
"""
from pathlib import Path

import pandas as pd
import numpy as np

from IPython import display
from jiwer import wer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchaudio.models.decoder._ctc_decoder import ctc_decoder
from torchnlp.encoders import LabelEncoder

import librosa
import matplotlib.pyplot as plt


from dataset import AN4Dataset, ProcessedDataset

"""
    Constants
"""
BATCH_SIZE = 8

"""
    Init Datasets
"""
path = Path("/Users/amitroth/PycharmProjects/ASR/an4")
train = AN4Dataset(path / Path("train"))
test = AN4Dataset(path / Path("test"))
val = AN4Dataset(path / Path("val"))

train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)


"""
    Create Vocab
"""
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
encoder = LabelEncoder(characters, reserved_labels=['unknown'], unknown_index=0)
print(f"Created Vocab: {encoder.vocab} (size: {len(encoder.vocab)}")


"""
    for debug
"""

def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    plt.show(block=False)


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    plt.show(block=False)



"""
    Encode torch array and string label into mel spec and output index
"""


def encode(audio: torch.Tensor, text: str, show: bool = False):
    N_FFT = 384
    HOP_LENGTH = 160
    WIN_LENGTH = 256

    # calculate spec
    spectrogram = torch.stft(audio.squeeze(0), n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    spectrogram = torch.sqrt((spectrogram ** 2).sum(-1))

    # normalize
    mean = torch.mean(spectrogram, 1, keepdim=True)
    std = torch.std(spectrogram, 1, keepdim=True)
    spectrogram = (spectrogram - mean) / (std + 1e-10)

    if show:
        fig, axs = plt.subplots(2, 1)
        plot_waveform(audio, train.sr, title="Original waveform", ax=axs[0])
        plot_spectrogram(spectrogram, title="spectrogram", ax=axs[1])
        fig.tight_layout()

    label = [encoder.encode(c) for c in text.lower()]

    return spectrogram, label

"""
    Define The model
"""
class FullyConnected(torch.nn.Module):
    """
    Args:
        n_feature: Number of input features
        n_hidden: Internal hidden unit size.
    """

    def __init__(self, n_feature: int, n_hidden: int, dropout: float, relu_max_clip: int = 20) -> None:
        super(FullyConnected, self).__init__()
        self.fc = torch.nn.Linear(n_feature, n_hidden, bias=True)
        self.relu_max_clip = relu_max_clip
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.hardtanh(x, 0, self.relu_max_clip)
        if self.dropout:
            x = torch.nn.functional.dropout(x, self.dropout, self.training)
        return x


class ASR(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rnn_layers: int = 1, rnn_units: int = 128):
        super().__init__()
        n_hidden = 200
        dropout = 0
        self.fc1 = FullyConnected(input_dim, n_hidden, dropout)
        self.bi_rnn = torch.nn.RNN(n_hidden, n_hidden, num_layers=rnn_layers, nonlinearity="relu", bidirectional=True)
        self.fc2 = FullyConnected(n_hidden, n_hidden, dropout)
        self.out = torch.nn.Linear(n_hidden, output_dim)

    def forward(self, x):
        x = self.fc1(x)        # N x C x T x F
        x = x.squeeze(1)        # N x C x T x H
        x = x.transpose(0, 1)        # N x T x H
        x, _ = self.bi_rnn(x)        # T x N x H
        x = x[:, :, : self.n_hidden] + x[:, :, self.n_hidden:]
        x = self.fc2(x)        # T x N x H
        x = self.out(x)        # T x N x H
        x = x.permute(1, 0, 2)        # T x N x n_class
        x = torch.nn.functional.log_softmax(x, dim=2)        # N x T x n_class
        return x        # N x T x n_class



"""
    Training loop
"""



if __name__ == '__main__':
    print(train_dataloader)

    sss = ProcessedDataset(path / Path("train"))

    train_features, train_labels = next(iter(train_dataloader))
    print(train_features)
    print(train_labels)





