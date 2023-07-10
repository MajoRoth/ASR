"""
    simple asr based on acoustic model and greedy ctc search
"""
from pathlib import Path

import torch
import torchaudio
from jiwer import wer
from tqdm import tqdm

from dataset import AN4Dataset
from ctc import GreedyCTCDecoder


class AcousticGreedyCTC:
    """
        A model that combines WAV2VEC and greedy ctc to make a prediction
    """

    def __init__(self):
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
        self.sr = bundle.sample_rate

        self.acoustic_model = bundle.get_model()
        self.tokens = [label.lower() for label in bundle.get_labels()]
        self.decoder = GreedyCTCDecoder(self.tokens)

    def __call__(self, *args, **kwargs):
        if len(args) != 1:
            raise Exception("you should supply a torch tenor as argument")

        audio = args[0]
        emission, _ = self.acoustic_model(audio)
        result = self.decoder(emission[0])
        return " ".join(result)

    def calculate_wer(self, wav, label):
        predicted = self.__call__(wav)
        return wer(label.lower(), predicted.lower())




if __name__ == "__main__":
    path = Path("/Users/amitroth/PycharmProjects/ASR/an4")
    val = AN4Dataset(path / Path("val"))

    model = AcousticGreedyCTC()

    for i in range(1):
        wav, label = val[i]
        print(f"PREDICTED: {model(wav)}  ---  LABELED: {label}")

    total_wer = 0
    print("calculating average WER")
    for wav, label in tqdm(val):
        total_wer += model.calculate_wer(wav, label)
    print(f"Average WER is {total_wer / len(val)}")





