from typing import List

import torch
from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState
from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder._ctc_decoder import download_pretrained_files

from dataset_preprocessed import EPSILON


class LanguageModelCTC(torch.nn.Module):

    LM_WEIGHT = 2.23
    WORD_SCORE = -0.26

    def __init__(self, labels, blank=0):
        self.labels = labels
        self.blank = blank

        files = download_pretrained_files("librispeech-4-gram")

        # with open("/Users/amitroth/PycharmProjects/ASR/data/tokens.txt", 'w') as f:
        #     for token in tokens:
        #         f.writelines(token)
        #         f.writelines("\n")
        # print(tokens)
        self.decoder = ctc_decoder(
            lexicon="/Users/amitroth/PycharmProjects/ASR/data/an4-vocab-lexicon.txt",
            tokens="/Users/amitroth/PycharmProjects/ASR/data/tokens.txt",
            lm=files.lm,
            nbest=3,
            beam_size=200,
            lm_weight=LanguageModelCTC.LM_WEIGHT,
            word_score=LanguageModelCTC.WORD_SCORE,
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

    def __str__(self):
        return "LanguageModelCTC"
