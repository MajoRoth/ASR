from model_getter import *
from confs.confs import AttrDict
from dataset_preprocessed import build_datasets, CharDictionary, TEXT_MIN_ASCII_VAL, TEXT_MAX_ASCII_VAL
import json

import argparse
from asr import ASR
from ctc import GreedyCTC, LexiconCTC, LanguageModelCTC
from jiwer import wer


def main(args):
    char_dict = CharDictionary(TEXT_MIN_ASCII_VAL, TEXT_MAX_ASCII_VAL)
    cfg = AttrDict(json.load(open(args.conf)))
    train_ds, val_ds = build_datasets(args, cfg)

    acoustic_models = load_all_models(args)
    ctc_models = [
        GreedyCTC(char_dict), LexiconCTC(char_dict), LanguageModelCTC(char_dict)
    ]

    print(f'ARGS: {args}')
    print(f'PARAMS: {cfg}')

    asr = ASR(acoustic_models[args.num_acoustic], ctc_models[args.num_ctc])
    print(f"constructing asr model {asr}")
    print(f"transcribing wav: {args.wav_path}")

    # load wav and transcribe

def get_parser():
  parser = argparse.ArgumentParser(description='try an asr model')
  parser.add_argument('--conf', type=str, required=True)
  parser.add_argument('--wav_path', type=str)
  parser.add_argument('--num_acoustic', type=int, default=0)
  parser.add_argument('--num_ctc', type=int, default=0)
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--ckpt_step', type=int, default=None)



  return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)