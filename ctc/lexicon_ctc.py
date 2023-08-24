import time
from pathlib import Path
from typing import List

import torch

from abc import ABC, abstractmethod

import torchaudio
from torchaudio.models.decoder._ctc_decoder import ctc_decoder
from tqdm import tqdm

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

    def create_lexicon(self, data_path: str = "/ctc/an4/train/", output_path: str = "/an4-vocab.txt"):
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

# class Route:
#
#     def __init__(self, init_list=[], probability=0):
#         self.route = init_list
#         self.probability = probability
#
#     def __le__(self, other):
#         if type(other) != type(self):
#             raise Exception("Invalid type for compression")
#         return self.probability <= other.probability
#
#     def get_next_route(self, index: int, probability: float):
#         return Route(self.route + [index], self.probability + probability)




class LexiconCTC(torch.nn.Module):

    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

        self.an4_lexicon = AN4Lexicon(lexicon_path="/Users/amitroth/PycharmProjects/ASR/data/an4-vocab.txt")
        self.librispeech_lexicon = LibrispeechLexicon(lexicon_path="/Users/amitroth/PycharmProjects/ASR/data/librispeech-vocab.txt")

        self.decoder = ctc_decoder(
            lexicon="/Users/amitroth/PycharmProjects/ASR/data/an4-vocab-lexicon.txt",
            tokens="/Users/amitroth/PycharmProjects/ASR/data/tokens.txt",
            nbest=3,
            beam_size=200,
        )



    def forward(self, emission: torch.Tensor) -> List[str]:
        output = list()
        result = self.decoder(emission)
        for i in result:
            tokens = i[0].tokens
            joined = self.labels.tokens2text(tokens)

            """
                drop spaces
            """
            output.append(joined.strip())
        return output



    def slow_forward(self, emission: torch.Tensor) -> List[str]:
        """
            Calculating the best k options
            choosing the sentence with best score

            VERY TIME CONSUMING, NEED TO CHECK DIFFERENT METHODS
        """
        output = list()
        for batch in emission:
            k_routes = [
                [list(), 0.0]  # each route contains the letters he visited and the current score
            ]
            for letter in tqdm(batch):
                letter = (letter - torch.min(letter) + 0.001) / (torch.max(letter) - torch.min(letter))  # normalize
                candidates = list()

                for route, score in k_routes:
                    for j in range(len(letter)):
                        new_score = score - torch.log(letter[j])

                        candidates.append(
                            [route + [j], new_score]
                        )
                sorted_candidates = sorted(candidates, key=lambda x: x[1])
                k_routes = sorted_candidates[:LexiconCTC.K]


            calculated_routes = set()
            for route, score in k_routes:
                indices = torch.unique_consecutive(torch.tensor(route), dim=-1)
                indices = [i for i in indices if i != self.blank]
                # joined = "".join([self.labels[i] for i in indices])
                # result = joined.replace("|", " ").strip()
                # output.add(result)

                joined = self.labels.tokens2text(indices)
                calculated_routes.add(joined)

            max_score = -1
            result = None
            for res in calculated_routes:
                score = 0
                for w in res.split():
                    if self.an4_lexicon.is_word(w):
                        score += 1
                    if self.librispeech_lexicon.is_word(w):
                        score += 0.1

                if score > max_score:
                    max_score = score
                    result = res

            output.append(result)

        return output

    def __str__(self):
        return "LexiconCTC"





