from torch import nn
import torchaudio


class LinearAcoustic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.acoustic_emb = nn.Linear(cfg.n_mels, cfg.n_tokens)
    
    def forward(self, x):
        """
        Inputs:
        x: melspec of shape (B, T, n_mels)
        """
        return self.acoustic_emb(x)