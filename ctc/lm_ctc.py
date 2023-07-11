from typing import List

import torch
from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState
from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder._ctc_decoder import download_pretrained_files


# @TODO curently imported from pytorch, do i need to change it? @Alon @Amit?

class CustomLM(CTCDecoderLM):
    """Create a Python wrapper around `language_model` to feed to the decoder."""

    def __init__(self, language_model: torch.nn.Module):
        CTCDecoderLM.__init__(self)
        self.language_model = language_model
        self.sil = -1  # index for silent token in the language model
        self.states = {}

        language_model.eval()

    def start(self, start_with_nothing: bool = False):
        state = CTCDecoderLMState()
        with torch.no_grad():
            score = self.language_model(self.sil)

        self.states[state] = score
        return state

    def score(self, state: CTCDecoderLMState, token_index: int):
        outstate = state.child(token_index)
        if outstate not in self.states:
            score = self.language_model(token_index)
            self.states[outstate] = score
        score = self.states[outstate]

        return outstate, score

    def finish(self, state: CTCDecoderLMState):
        return self.score(state, self.sil)


class LanguageModelCTC(torch.nn.Module):

    LM_WEIGHT = 3.23
    WORD_SCORE = -0.26

    def __init__(self):
        files = download_pretrained_files("librispeech-4-gram")

        self.decoder = ctc_decoder(
            lexicon=files.lexicon,
            tokens=files.tokens,
            lm=files.lm,
            nbest=3,
            beam_size=1500,
            lm_weight=LanguageModelCTC.LM_WEIGHT,
            word_score=LanguageModelCTC.WORD_SCORE,
        )

    def forward(self, emission: torch.Tensor) -> List[str]:
        result = self.decoder(emission)
        transcript = " ".join(result[0][0].words).strip()
        return transcript
