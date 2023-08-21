import json

from models.linear_acoustic_model import LinearAcoustic
from models.DeepSpeech2 import DS2LargeModel, DS2SmallModel, DS2ToyModel
from models.mini_lstm_model import MiniLSTM
import torch
from pathlib import Path
from confs.confs import AttrDict



def get_model(args=None, cfg=None):
    if cfg.model == "LinearAcoustic":
        return LinearAcoustic(cfg)
    elif cfg.model == 'DeepSpeech2_Large':
        return DS2LargeModel(cfg)
    elif cfg.model == 'DeepSpeech2_Small':
        return DS2SmallModel(cfg)
    elif cfg.model == 'DeepSpeech2_Toy':
        return DS2ToyModel(cfg)
    elif cfg.model == 'MiniLSTM':
        return MiniLSTM(cfg)
    else:
        raise NotImplementedError

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
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"device is {device}")
        checkpoint = torch.load(ckpt_filename, map_location=torch.device(device))
        model, step = load_state_dict(model, checkpoint)
        print("Loaded {} from {} step checkpoint".format(f'{model_dir}/{cfg.model}_{ckpt_type}.pt', step))
        return model, step

def load_all_models(args):
    # load all 3 models with best ckpts and resturn them as list

    ckpts_dir = "checkpoints"

    models = list()
    for conf in Path("confs").glob("*.json"):
        cfg = AttrDict(json.load(open(str(conf))))
        model_dir = f"{ckpts_dir}/{cfg.run_name}"
        model = get_model(cfg=cfg)
        model, step = restore_from_checkpoint(model, model_dir, step=args.ckpt_step, cfg=cfg, ckpt_type='best')
        models.append(model)

    return models

