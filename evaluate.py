from model_getter import *
from confs.confs import AttrDict
from dataset_preprocessed import build_datasets, CharDictionary, TEXT_MIN_ASCII_VAL, TEXT_MAX_ASCII_VAL
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import argparse
from asr import ASR
from ctc import GreedyCTC, LexiconCTC, LanguageModelCTC
from jiwer import wer

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def draw_graphs():

    # matplotlib settings
    # mpl.rcParams['font.family'] = "Arial"
    # mpl.rcParams['font.weight'] = "bold"
    # plt.rcParams['font.size'] = 22
    # plt.rcParams['axes.linewidth'] = 3

    species = ("Linear", "LSTM", "DeepSpeech\nToy", "DeepSpeech\nSmall", "DeepSpeech\nLarge")
    data = {
        'Greedy': (0.99, 0.83, 0.33, 0, 0),
        'Lexicon': (0.98, 0.72, 0.21, 0, 0),
        'LM': (0.98, 0.77, 0.21, 0, 0)
    }

    augmented_data = {
        'Greedy': (0.99, 0.49, 0.2, 0, 0),
        'Lexicon': (0.99, 0.35, 0.16, 0, 0),
        'LM': (0.98, 0.35, 0.16, 0, 0)
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in augmented_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fontsize=8)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('WER')
    ax.set_title('WER by model measured augmented data')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper right')
    # ax.set_ylim(0, 250)

    plt.savefig("data/static/augmented_benchmarks")

def evaluate_single_model(asr: ASR, ds):
    """
    calculate wer
    """
    print(f"evaluating: {asr}")
    total_wer = list()

    output_text = list()
    for feats in ds:
        output = asr.transcribe(feats['x'])
        for i, prediction in enumerate(output):
            print(f"predicted: {prediction}")
            output_text.append(prediction)
            print(f"GT: {feats['text'][i]}")
            word_error = wer(feats['text'][i], prediction)

            total_wer.append(word_error)
            print(f"wer {word_error}")


    print(f"Finished evaluating. Total error: {bcolors.OKCYAN}{sum(total_wer) / len(total_wer)}{bcolors.ENDC}")
    return output_text


def main(args):
    char_dict = CharDictionary(TEXT_MIN_ASCII_VAL, TEXT_MAX_ASCII_VAL)
    cfg = AttrDict(json.load(open(args.conf)))
    print(cfg)
    train_ds, test_ds, val_ds = build_datasets(args, cfg)

    print(f'ARGS: {args}')
    print(f'PARAMS: {cfg}')

    acoustic_models = load_all_models(args)
    ctc_models = [
        GreedyCTC(char_dict), LexiconCTC(char_dict), LanguageModelCTC(char_dict)
    ]
    predictions = list()

    print("--- val ---")
    for acoustic_model in acoustic_models:
        for ctc_model in ctc_models:

            asr = ASR(acoustic_model, ctc_model)
            predictions.append(evaluate_single_model(asr, val_ds))

    print("--- test ---")
    for acoustic_model in acoustic_models:
        for ctc_model in ctc_models:
            asr = ASR(acoustic_model, ctc_model)
            predictions.append(evaluate_single_model(asr, test_ds))

    print("--- train ---")
    for acoustic_model in acoustic_models:
        for ctc_model in ctc_models:

            asr = ASR(acoustic_model, ctc_model)
            predictions.append(evaluate_single_model(asr, train_ds))

    # print("--- diff ---")
    # for i in range(len(predictions[0])):
    #     diff = list()
    #     for j in range(len(predictions)):
    #         diff.append(predictions[j][i])
    #     print(diff)


# def main(args):
#   cfg = AttrDict(
#     json.load(open(args.conf)))
#
#   print(f'ARGS: {args}')
#   print(f'PARAMS: {cfg}')
#   ckpts_dir = "checkpoints"
#   model_dir = f"{ckpts_dir}/{cfg.run_name}"
#
#   model = get_model(args, cfg)
#   model, step = restore_from_checkpoint(model, model_dir, step=args.ckpt_step, cfg=cfg, ckpt_type='best')
#
#   ctc = GreedyCTC(model.token_dict)
#   asr = ASR(model, ctc)
#
#
#   train_ds, val_ds = build_datasets(args, cfg)
#
#   N_BATCHES_TO_EVAL = 2
#   for feats in val_ds:
#       output = asr.transcribe(feats['x'])
#       print(" --- batch ---")
#       print(output)
#       print(feats['text'])
#       probs = model.predict(feats['x'])
#       pred_text = model.greedy_transcription(probs)
#       GT_text = model.tokens_to_text(feats['label'])
#
#       for pred_t, GT_t in zip(pred_text, GT_text):
#           print(f"GT text: {GT_t}  pred text: {pred_t}")
#
#       # text_from_wavs = model.wavs_to_greedy_transcription(feats['wav'])
#       # print(f'text_from_probs: {text_from_probs}  text_from_wavs: {text_from_wavs}')
#
#       # probs_from_wav = model.predict_from_wav(feats['wav'])
#       # assert probs.shape == probs_from_wav.shape


def get_parser():
  parser = argparse.ArgumentParser(description='evaluate all asr models')
  parser.add_argument('--conf', type=str, required=True)
  parser.add_argument('--ckpt_step', type=int, default=None)
  parser.add_argument('--num_workers', type=int, default=4)

  return parser


if __name__ == "__main__":
    draw_graphs()
    # parser = get_parser()
    # args = parser.parse_args()
    # main(args)
