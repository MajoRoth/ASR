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




if __name__ == '__main__':

    path = Path("/Users/amitroth/PycharmProjects/ASR/an4")

    train = AN4Dataset(path / Path("train"))
    test = AN4Dataset(path / Path("test"))
    val = AN4Dataset(path / Path("val"))

    print(train[0])

