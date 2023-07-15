
from torch.cuda import device_count
from torch.multiprocessing import spawn
from confs.confs import AttrDict

import torch
import json
import argparse

import os

from model_getter import *
from train_impl import train

def main(args):
  cfg = AttrDict(
    json.load(open(args.conf)))

  print(f'ARGS: {args}')
  print(f'PARAMS: {cfg}')
  ckpts_dir = "checkpoints"
  args.model_dir = f"{ckpts_dir}/{cfg.run_name}"
  os.makedirs(args.model_dir, exist_ok=True)
  train(args, cfg)

def get_parser():
  parser = argparse.ArgumentParser(description='train an AudioMPD model')
  parser.add_argument('--conf', type=str, required=True)
  parser.add_argument('--logger', type=str, choices=['wandb', 'tensorboard', 'none'], default='none')
  parser.add_argument('--num_workers', type=int, default=4)
  return parser

if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)