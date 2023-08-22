import os
from pathlib import Path

import torch
# from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchaudio
#from torchnlp.encoders import LabelEncoder

TEXT_MIN_ASCII_VAL = 32
TEXT_MAX_ASCII_VAL = 90
EPSILON = "|"


class CharDictionary():
    def __init__(self, text_min_ascii, text_max_asii):
        self.chr2label = {chr(ascii_val): (ascii_val - text_min_ascii + 1) for ascii_val in range(text_min_ascii, text_max_asii+1)}
        self.label2chr = {v: k for k,v in self.chr2label.items()}
        self.label2chr[0] = EPSILON

    def text2tokens(self, text):
        return torch.tensor([self.chr2label[c] for c in text], dtype=int)

    def tokens2text(self, labels):
        return "".join([self.label2chr[int(l)] for l in labels])

class Wav2MelSpec():
    def __init__(self, cfg):
        self.n_mels = cfg.n_mels

    def __call__(self, wav, cuda=False):
        if cuda:
            to_mel = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels).cuda()
        else:
            to_mel = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels)

        melspec = to_mel(wav)
        if len(melspec.shape) == 2:
            # [n_mels, seq_len]
            return melspec.T
        else:
            # [batch, n_mels, seq_len]
            return melspec.permute(0, 2, 1)
            

class AN4DatasetPreprocessed(Dataset):
    """
        Inherit from a PyTorch Dataset to support dataloader
    """

    def __init__(self, path, cfg, est_norm_factors):
        if not os.path.exists(path):
            raise Exception(f"Invalid Path {path}")
        
        self.cfg = cfg
        self.wav2mel = Wav2MelSpec(cfg)
        self.est_norm_factors = est_norm_factors

        self.path = Path(path)
        self.data = list()

        self.text_dir_path = self.path.rglob("*.txt")
        self.wav_dir_path = self.path.rglob("*.wav")
        
        self.parse_files()
        
        self.token_dict = CharDictionary(TEXT_MIN_ASCII_VAL, TEXT_MAX_ASCII_VAL)        
    
    def parse_files(self):
        self.zipped_paths = list(zip(sorted(self.wav_dir_path), sorted(self.text_dir_path)))
        self.max_seq_len = 0
        self.max_text_len = 0
        self.max_wav_len = 0
        self.text_min_ascii = 100000
        self.text_max_ascii = -1    

        melspecs_pool = None
        MAX_FRAMES_FOR_STATS = 100000

        for tup in self.zipped_paths:
            assert tup[0].stem == tup[1].stem
        for wav_path, text_path in self.zipped_paths:
            assert text_path.stem == wav_path.stem

            wav, sr = torchaudio.load(wav_path)
            assert wav.shape[0] == 1 # assert mono
            wav = wav[0]
            melspec = self.wav2mel(wav) # time, n_mels
            
            if self.est_norm_factors:
                if melspecs_pool is None:
                    melspecs_pool = melspec
                elif len(melspecs_pool) < MAX_FRAMES_FOR_STATS:
                    melspecs_pool = torch.concat((melspecs_pool, melspec), dim=0)


            self.max_seq_len = max(self.max_seq_len, len(melspec))
            self.max_wav_len = max(self.max_wav_len, len(wav))


            with open(text_path) as f:
                text = f.readline()
                text_ascii = [ord(c) for c in text]
            self.max_text_len = max(self.max_text_len, len(text))

            self.text_min_ascii = min(self.text_min_ascii, min(text_ascii))
            self.text_max_ascii = max(self.text_max_ascii, max(text_ascii))

            self.data.append(
                (wav, melspec, text)
            )
        self.sr = sr
        
        assert self.text_min_ascii >= TEXT_MIN_ASCII_VAL and self.text_max_ascii <= TEXT_MAX_ASCII_VAL
        if self.est_norm_factors:
            self.calc_normalization_factors(melspecs_pool)
        else:
            self.feats_std = torch.ones((1, self.cfg.n_mels))
            self.feats_mean = torch.zeros((1, self.cfg.n_mels))
    
    def calc_normalization_factors(self, melspecs_pool):
        self.feats_std = torch.clamp(melspecs_pool.std(dim=0, keepdim=True), min=0.1, max=10.0)
        self.feats_mean = torch.clamp(melspecs_pool.mean(dim=0, keepdim=True), min=-10.0, max=10.0)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        wav, melspec, text = self.data[idx]

        # per feature normalization 
        if self.cfg.normalize_features:
            melspec = (melspec - self.feats_mean) / self.feats_std  

        # zero padding
        n = len(melspec)
        if n < self.max_seq_len:
            melspec = torch.concat((melspec, torch.zeros((self.max_seq_len - n, self.cfg.n_mels))))
        n_wav = len(wav)
        if n_wav < self.max_wav_len:
            wav = torch.concat((wav, torch.zeros(self.max_wav_len - n_wav)))
        label = self.token_dict.text2tokens(text)

        text_len = len(label)
        if text_len < self.max_text_len:
            label = torch.concat((label, torch.zeros(self.max_text_len - text_len, dtype=int))) 
        return {"x": melspec, "length": n, "label": label, "label_length": text_len, "wav": wav, "text": text}


    def __str__(self):
        return f"Dataset: {self.path}, length: {self.__len__()} files"


    def __repr__(self):
        return self.__str__()


def build_single_dataset(ds_path, args, cfg, is_training):
    dataset = AN4DatasetPreprocessed(ds_path, cfg, est_norm_factors=is_training)
    shuffle = is_training and cfg.shuffle_train_ds
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        sampler=None,
        pin_memory=True,
        drop_last=is_training)

def build_datasets(args, cfg):
    datasets_path = Path(cfg.data_dir)
    train_ds = build_single_dataset(datasets_path / Path("train"), args, cfg, is_training=True)
    val_ds = build_single_dataset(datasets_path / Path("val"), args, cfg, is_training=False)
    test_ds = build_single_dataset(datasets_path / Path("test"), args, cfg, is_training=False)

    val_ds.dataset.feats_std = train_ds.dataset.feats_std
    val_ds.dataset.feats_mean = train_ds.dataset.feats_mean
    val_ds.dataset.token_dict = train_ds.dataset.token_dict

    test_ds.dataset.feats_std = train_ds.dataset.feats_std
    test_ds.dataset.feats_mean = train_ds.dataset.feats_mean
    test_ds.dataset.token_dict = train_ds.dataset.token_dict

    # n_tokens_train = train_ds.dataset.text_max_ascii - train_ds.dataset.text_min_ascii 
    # n_tokens_val = val_ds.dataset.text_max_ascii - val_ds.dataset.text_min_ascii 
    # assert train_ds.dataset.text_min_ascii == val_ds.dataset.text_min_ascii
    # n_tokens = max(n_tokens_train, n_tokens_val) + 1 # adding one for the blank character
    # print(f'n_tokens: {n_tokens}')

    return train_ds, test_ds, val_ds


def construct_lexicon():
    cache_dict = "/Users/amitroth/PycharmProjects/ASR/data/an4-vocab.txt"
    new_dict = "/Users/amitroth/PycharmProjects/ASR/data/an4-vocab-lexicon.txt"
    with open(cache_dict, 'r') as cache:
        with open(new_dict, 'w') as new:
            for line in cache.readlines():
                letters = " ".join([c for c in line])
                new.writelines(f"{line[:-1]} {letters[:-1]} |\n".upper())


if __name__ == '__main__':
    construct_lexicon()
    #
    # path = Path("/Users/amitroth/PycharmProjects/ASR/an4")
    #
    # train = AN4Dataset(path / Path("train"))
    # test = AN4Dataset(path / Path("test"))
    # val = AN4Dataset(path / Path("val"))
    #
    # print(train[0])



