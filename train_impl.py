import os
import torch
from torch.nn.parallel import DistributedDataParallel
from dataset_preprocessed import build_datasets
from model_getter import *
from trainers.acoustic_trainer import AcousticTrainer
import wandb


def print_num_params(model, model_name):
    print(f"{model_name} has {sum(p.numel() for p in model.parameters())} params")
    print(f"    {sum(p.numel() for p in model.parameters() if p.requires_grad)} of which are trainable")


def get_optimizer(model, cfg):
    return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)


def _train_impl(model, dataset, dataset_val, args, cfg):
    torch.backends.cudnn.benchmark = True
    opt = get_optimizer(model, cfg)
    trainer_class = AcousticTrainer
    trainer = trainer_class(args, model, dataset, dataset_val, opt, cfg)
    trainer.is_master = True
    trainer.restore_from_checkpoint()
    trainer.train(max_steps=cfg.max_steps)


def get_wandb_id(wandb_id_file):
    if os.path.exists(wandb_id_file):
        with open(wandb_id_file, 'r') as f:
            return f.readline()
        return None


def init_logger_if_needed(args, cfg):
    if args.logger == "wandb":
        wandb_id_file = f"{args.model_dir}/wandb_id.txt"
        wandb_run_id = get_wandb_id(wandb_id_file)
        if wandb_run_id is not None:
            wandb.init(project="ASR_Proj", name=cfg.run_name,
                       dir=f"{args.model_dir}", resume="allow", id=wandb_run_id)
            assert wandb_run_id == wandb.run.id
            print(f"resuming run with wandb unique id: {wandb.run.id}")

        else:
            wandb.init(project="ASR_Proj", name=cfg.run_name,
                       dir=f"{args.model_dir}", resume="allow")
            with open(f"{args.model_dir}/wandb_id.txt", 'w') as f:
                f.write(f'{wandb.run.id}')
                print(f"created new wandb unique id: {wandb.run.id}")


def train(args, cfg):
    init_logger_if_needed(args, cfg)

    dataset_train, _, dataset_val = build_datasets(args, cfg)

    model = get_model(args, cfg)
    if torch.cuda.is_available():
        model = model.cuda()

    print_num_params(model, "model")

    _train_impl(model, dataset_train, dataset_val, args, cfg)
