from torch import nn
import torch
import torchaudio

from dataset_preprocessed import EPSILON, TEXT_MIN_ASCII_VAL, TEXT_MAX_ASCII_VAL, CharDictionary, Wav2MelSpec
import re

class LinearAcoustic(nn.Module):
    def __init__(self, cfg, token_dict):
        super().__init__()
        self.cfg = cfg
        self.acoustic_emb = nn.Linear(cfg.n_mels, cfg.n_tokens)
        self.token_dict = CharDictionary(TEXT_MIN_ASCII_VAL, TEXT_MAX_ASCII_VAL)
        self.to_melspec = Wav2MelSpec(cfg)

    
    def forward(self, x):
        """
        Inputs:
        x: melspec of shape (B, T, n_mels)
        """
        return self.acoustic_emb(x)
    
    def predict_from_wav(self, wav):
        return self.predict(self.to_melspec(wav))

    def predict(self, x):
        """
        Inputs:
        x: melspec of shape (B, T, n_mels)
        """
        logits = self.forward(x)
        return logits.softmax(dim=-1)
    
    def tokens_to_text(self, tokens):
        texts = [self.token_dict.tokens2text(tokens_row) for tokens_row in tokens]
        texts = [re.sub(EPSILON, '', t) for t in texts] 
        return texts

    def greedy_transcription(self, probs):
        tokens = torch.argmax(probs, dim=-1)
        text_non_collapsed = self.tokens_to_text(tokens)
        text = ["".join(dict.fromkeys(t)) for t in text_non_collapsed]
        return text

    def wavs_to_greedy_transcription(self, wavs):
        return self.greedy_transcription(self.predict_from_wav(wavs))


