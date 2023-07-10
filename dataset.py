import os
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchaudio
from torchnlp.encoders import LabelEncoder



class AN4Dataset(Dataset):
    """
        Inherit from a PyTorch Dataset to support dataloader
    """

    def __init__(self, path):
        if not os.path.exists(path):
            raise Exception(f"Invalid Path {path}")

        self.path = Path(path)
        self.data = list()

        self.text_dir_path = self.path.rglob("*.txt")
        self.wav_dir_path = self.path.rglob("*.wav")

        self.zipped_paths = list(zip(sorted(self.wav_dir_path), sorted(self.text_dir_path)))

        for tup in self.zipped_paths:
            assert tup[0].stem == tup[1].stem

        for wav_path, text_path in self.zipped_paths:
            assert text_path.stem == wav_path.stem

            wav, sr = torchaudio.load(wav_path)
            with open(text_path) as f:
                text = f.readline()

            self.data.append(
                (wav, text)
            )

        self.sr = sr

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]


    def __str__(self):
        return f"Dataset: {self.path}, length: {self.__len__()} files"


    def __repr__(self):
        return self.__str__()
    

class ProcessedDataset(AN4Dataset):
    
    def __init__(self, path):
        super().__init__(path)

        self.characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
        self.encoder = LabelEncoder(self.characters, reserved_labels=['unknown'], unknown_index=0)

        self.processed_data = list()
        for audio, text in self.data:
            spectrogram, label = encode(audio, text, self.encoder)
            self.processed_data.append((spectrogram, label))

            # print(audio.shape)
            # print(spectrogram.shape)
            # print(audio.shape[1] / spectrogram.shape[1])
            # print("\n")


    def __getitem__(self, idx):
        return self.processed_data[idx]



def encode(audio: torch.Tensor, text: str, encoder):
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

    label = [encoder.encode(c) for c in text.lower()]

    return spectrogram, label














if __name__ == '__main__':

    path = Path("/Users/amitroth/PycharmProjects/ASR/an4")

    train = AN4Dataset(path / Path("train"))
    test = AN4Dataset(path / Path("test"))
    val = AN4Dataset(path / Path("val"))

    sss = ProcessedDataset(path / Path("train"))

    print(train[0])

