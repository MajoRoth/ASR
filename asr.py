from torch import nn


class ASR:
    def __init__(self, acoustic_model: nn.Module, ctc_model: nn.Module):
        self.acoustic_model = acoustic_model
        self.ctc_model = ctc_model

    def transcribe(self, x):
        """
            Inputs:
            x: melspec of shape (B, T, n_mels)
        """

        acoustic_output = self.acoustic_model.forward(x)
        text = self.ctc.forward(acoustic_output)
        return text

    def __str__(self):
        return f"ASR with acoustic model: {self.acoustic_model}   and ctc model: {self.ctc_model}"
