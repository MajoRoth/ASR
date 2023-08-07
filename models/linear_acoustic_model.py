from models.base_acoustic_model import BaseAcousticModel
from torch import nn

class LinearAcoustic(BaseAcousticModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.acoustic_emb = nn.Linear(cfg.n_mels, cfg.n_tokens)

    def forward(self, x):
        """
        Inputs:
        x: melspec of shape (B, T, n_mels)
        """
        return self.acoustic_emb(x)
    