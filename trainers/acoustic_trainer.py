
from torch.utils.tensorboard import SummaryWriter
from trainers.generic_trainer import GenericTrainer, _nested_map
from tqdm import tqdm

import torch.nn as nn
import torch
import wandb
import numpy as np
import librosa


class AcousticTrainer(GenericTrainer):
    def __init__(self, args, model, dataset, dataset_val, opt, cfg):
        super().__init__(args, cfg, model, dataset, opt)
        self.dataset_val = dataset_val
        self.name = cfg.model
        self.logger = args.logger
        self.cfg = cfg
        self.lr = cfg.learning_rate
        self.summary_writer = None
        self.loss = nn.CTCLoss()

    def forward_and_loss(self, features):
        logits = self.model(features['x']) # B, T, C
        logsoftmax = torch.nn.functional.log_softmax(logits.permute(1, 0, 2), dim=-1)
        loss = self.loss(logsoftmax, features['label'], features['length'], features['label_length'])
        return {"logits": logits, "target": features['label'], "loss": loss}

    def train_step(self, features, max_steps):
        if self.mixed_percision_training:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.forward_and_loss(features)
        else:
            outputs = self.forward_and_loss(features)

        loss = outputs["loss"]

        # backward and update
        loss = self.backward_and_update(loss, max_steps)

        return loss

    def run_valid_loop(self, MAX_VAL_BATCHES=300):
        print("INFO: Running validation loop")
        losses = []
   
        with torch.no_grad():
            device = next(self.model.parameters()).device

            for i, features in enumerate(tqdm(self.dataset_val, desc=f'Valid {len(self.dataset_val)}')):

                features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)

                outputs = self.forward_and_loss(features)

                losses.append(outputs["loss"].cpu().numpy())

                if i >= MAX_VAL_BATCHES:
                    break

            loss = np.mean(losses)

            self._write_summary_valid(self.step, loss)

    def _write_summary(self, step, features, loss):
        if self.logger == "tensorboard":
            writer = self.summary_writer or SummaryWriter(f"{self.model_dir}/tensorboard", purge_step=step)
            writer.add_scalar('train/loss', loss * self.grad_accum_steps, step)
            writer.add_scalar('opt/learning_rate', self.lr, step)
            writer.add_scalar('opt/grad_norm', self.grad_norm, step)
            writer.flush()
            self.summary_writer = writer

        if self.logger == "wandb":
            wandb.log({"train/loss": loss * self.grad_accum_steps}, step=step)
            wandb.log({"train/learning_rate": self.lr}, step=step)
            wandb.log({"train/grad_norm": self.grad_norm}, step=step)

    def _write_summary_valid(self, step, loss):
                             
        if self.logger == "tensorboard":
            writer = self.summary_writer or SummaryWriter(f"{self.model_dir}/tensorboard", purge_step=step)
            writer.add_scalar('val/loss', loss, step)
            writer.flush()
            self.summary_writer = writer

        if self.logger == "wandb":
            wandb.log({"validation/loss": loss}, step=step)