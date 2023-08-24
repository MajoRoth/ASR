from models.model_getter import *
from confs.confs import AttrDict
from dataset_preprocessed import build_datasets, CharDictionary, TEXT_MIN_ASCII_VAL, TEXT_MAX_ASCII_VAL
import json

import argparse
from asr import ASR
from ctc import GreedyCTC, LexiconCTC, LanguageModelCTC
from jiwer import wer

def get_asr_by_input(args):
    char_dict = CharDictionary(TEXT_MIN_ASCII_VAL, TEXT_MAX_ASCII_VAL)

    acoustic_models = load_all_models(args)
    ctc_models = [
        GreedyCTC(char_dict), LexiconCTC(char_dict), LanguageModelCTC(char_dict)
    ]

    print(f"Choose the following acoustic model by index [1-{len(acoustic_models)}]")
    for i in range(len(acoustic_models)):
        print(f"{i + 1}) {acoustic_models[i]}")

    try:
        acoustic_index = int(input()) - 1
    except Exception:
        print("illegal input")
        exit()

    print(f"Choose the following ctc by index [1-{len(ctc_models)}]")
    for i in range(len(ctc_models)):
        print(f"{i + 1}) {ctc_models[i]}")

    try:
        ctc_index = int(input()) - 1
    except Exception:
        print("illegal input")
        exit()

    return ASR(acoustic_models[acoustic_index], ctc_models[ctc_index])


def get_wav_by_input(args):
    cfg = AttrDict(json.load(open(args.conf)))

    datasets = build_datasets(args, cfg)

    print(f"Choose the following dataset index [1-{len(datasets)}]")
    for i in range(len(datasets)):
        print(f"{i + 1}) {datasets[i].dataset.path}")

    try:
        dataset_index = int(input()) - 1
    except Exception:
        print("illegal input")
        exit()

    print(f"Choose the following wav index [1-{len(datasets)}]")
    wavs = list()
    for i, feats in enumerate(datasets[dataset_index]):
        wavs.append(feats)
        print(f'{i + 1}) {feats["text"][0]}')

    try:
        wav_index = int(input()) - 1
    except Exception:
        print("illegal input")
        exit()

    return wavs[wav_index]['x']



def main(args):

    asr = get_asr_by_input(args)

    wav = get_wav_by_input(args)

    print(f"Transcribing {wav} with {asr}")
    output = asr.transcribe(wav)
    print(output)







    # asr = ASR(acoustic_models[args.num_acoustic], ctc_models[args.num_ctc])
    # print(f"constructing asr model {asr}")
    # print(f"transcribing wav: {args.wav_path}")

    # load wav and transcribe

def get_parser():
  parser = argparse.ArgumentParser(description='try an asr model')
  parser.add_argument('--conf', type=str, required=True)
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--ckpt_step', type=int, default=None)
  return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)