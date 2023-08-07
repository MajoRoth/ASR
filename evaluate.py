from model_getter import *
from confs.confs import AttrDict
from train_impl import train
from dataset_preprocessed import build_datasets
import json
import torch
import argparse

#############################################################################################
#### Taken from https://github.com/microsoft/NeuralSpeech/tree/master/PriorGrad-vocoder #####
def load_state_dict(model, state_dict):
    if hasattr(model, 'module') and isinstance(model.module, torch.nn.Module):
        model.module.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict['model'])
    step = state_dict['step']
    return model, step

def restore_from_checkpoint(model, model_dir, step, cfg, ckpt_type='best'):
    try:
        checkpoint = torch.load(f'{model_dir}/{cfg.model}-{step}.pt')
        model, step = load_state_dict(model, checkpoint)
        print("Loaded {}".format(f'{model_dir}/{cfg.model}-{step}.pt'))
        return model, step
    except FileNotFoundError:
        ckpt_filename = f'{model_dir}/{cfg.model}_{ckpt_type}.pt'
        print(f"Trying to load {ckpt_filename}...")
        checkpoint = torch.load(ckpt_filename)
        model, step = load_state_dict(model, checkpoint)
        print("Loaded {} from {} step checkpoint".format(f'{model_dir}/{cfg.model}_{ckpt_type}.pt', step))
        return model, step
#############################################################################################

def main(args):
  cfg = AttrDict(
    json.load(open(args.conf)))

  print(f'ARGS: {args}')
  print(f'PARAMS: {cfg}')
  ckpts_dir = "checkpoints"
  model_dir = f"{ckpts_dir}/{cfg.run_name}"
   
  model = get_model(args, cfg)
  model, step = restore_from_checkpoint(model, model_dir, step=args.ckpt_step, cfg=cfg, ckpt_type='best')

  train_ds, val_ds = build_datasets(args, cfg)

  N_BATCHES_TO_EVAL = 2
  for feats in val_ds:
      probs = model.predict(feats['x'])
      pred_text = model.greedy_transcription(probs)
      GT_text = model.tokens_to_text(feats['label'])
      
      for pred_t, GT_t in zip(pred_text, GT_text):  
          print(f"GT text: {GT_t}  pred text: {pred_t}")

      # text_from_wavs = model.wavs_to_greedy_transcription(feats['wav'])
      # print(f'text_from_probs: {text_from_probs}  text_from_wavs: {text_from_wavs}')

      # probs_from_wav = model.predict_from_wav(feats['wav'])
      # assert probs.shape == probs_from_wav.shape

def get_parser():
  parser = argparse.ArgumentParser(description='train an AudioMPD model')
  parser.add_argument('--conf', type=str, required=True)
  parser.add_argument('--ckpt_step', type=int, default=None)
  parser.add_argument('--num_workers', type=int, default=4)

  return parser

if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)