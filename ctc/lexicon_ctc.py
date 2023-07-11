from pathlib import Path
from typing import List

import torch

from abc import ABC, abstractmethod

import torchaudio

from dataset import AN4Dataset


class Lexicon(ABC):

    def __init__(self, lexicon_path: str):
        self.path = lexicon_path

    @abstractmethod
    def is_word(self, word: str):
        pass


class LibrispeechLexicon(Lexicon):

    def __init__(self, lexicon_path: str = "./../librispeech-vocab.txt"):
        super().__init__(lexicon_path)

        with open(self.path, 'r') as lexicon:
            self.words_set = {s.lower().strip() for s in lexicon.readlines()}

    def is_word(self, word: str):
        return word in self.words_set


class AN4Lexicon(Lexicon):

    def __init__(self, lexicon_path: str = None):
        self.words_set = set()

        if lexicon_path is None:
            print("specify a lexicon path or call to create_lexicon()")

        else:
            super().__init__(lexicon_path)

            with open(self.path, 'r') as lexicon:
                self.words_set = {s.lower().strip() for s in lexicon.readlines()}


    def create_lexicon(self, data_path: str = "./../an4/train/", output_path: str = "./../an4-vocab.txt"):
        text_paths = Path(data_path).rglob("*.txt")

        for text_path in text_paths:
            with open(str(text_path), 'r') as txt:
                line = txt.readline().lower().strip()
                for word in line.split():
                    self.words_set.add(word)

        with open(output_path, 'w') as lexicon:
            for word in self.words_set:
                lexicon.write(word + "\n")




    def is_word(self, word: str):
        return word in self.words_set


class LexiconCTC(torch.nn.Module):

    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

        self.an4_lexicon = AN4Lexicon(data_path="./../an4-vocab.txt")
        self.librispeech_lexicon = LibrispeechLexicon(data_path="./../librispeech-vocab.txt")

    def forward(self, emission: torch.Tensor) -> List[str]:
        """
            Calculating the best k options
            choosing the sentence with best score
        """

        raise NotImplemented()

        print(emission)
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]

        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()



# if __name__ == '__main__':
#     bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
#     bundle.get_model()
#     acoustic_model = bundle.get_model()
#
#     path = Path("/Users/amitroth/PycharmProjects/ASR/an4")
#     val = AN4Dataset(path / Path("val"))
#
#     wav, label = val[0]
#     emission, _ = acoustic_model(wav)
#     print(emission)
#     ctc = LexiconCTC(labels=[label.lower() for label in bundle.get_labels()])
