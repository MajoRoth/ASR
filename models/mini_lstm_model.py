"""LSTM layers module."""

from torch import nn
from models.base_acoustic_model import BaseAcousticModel

#################################################################################################################
######## TAKEN FROM https://github.com/facebookresearch/encodec/blob/main/encodec/modules/lstm.py ###############
#################################################################################################################
class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y
#################################################################################################################


class MiniLSTM(BaseAcousticModel):
    def __init__(self, cfg, hidden_dim=128):
        super().__init__(cfg)
        self.input_emb = nn.Linear(cfg.n_mels, hidden_dim)
        self.lstm = SLSTM(dimension=hidden_dim)
        self.to_logits = nn.Linear(hidden_dim, cfg.n_tokens)
    
    def forward(self, x):
        """
        Inputs:
        x: melspec of shape (B, T, n_mels)
        """
        x = self.input_emb(x)
        x = self.lstm(x.permute(0, 2, 1)).permute(0, 2, 1)
        logits = self.to_logits(x)
        return logits 