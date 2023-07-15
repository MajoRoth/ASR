import torch.nn as nn
import torch
import os
from tqdm import tqdm

##############################################################################################
##### Taken from https://github.com/microsoft/NeuralSpeech/tree/master/PriorGrad-vocoder #####
def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)
##############################################################################################

class GenericTrainer:
    def __init__(self, args, cfg, model, dataset, opt):
        self.model_dir = args.model_dir
        self.model = model
        self.dataset = dataset
        self.step = 0 # opt updates counter
        self.substep = 0 # including grad accum steps
        self.params = cfg
        self.optimizer = opt
        self.grad_norm = 0.0
        self.grad_accum_steps = cfg.grad_accum_steps
        self.mixed_percision_training = cfg.mixed_percision_training
        self.valid_loop_every = cfg.valid_loop_every
        self.ckpt_every = cfg.checkpoint_every

    ######################################################################################################
    ## Based on code taken from https://github.com/microsoft/NeuralSpeech/tree/master/PriorGrad-vocoder ##
    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'step': self.step,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in
                          self.optimizer.state_dict().items()},
            'params': dict(self.params),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{self.name}_{filename}-{self.step}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{self.name}_{filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            ckpt_file = f'{self.model_dir}/{self.name}_{filename}.pt'
            checkpoint = torch.load(ckpt_file)
            self.load_state_dict(checkpoint)
            print(f"INFO: model was restored from checkpoint {ckpt_file}")
            return True
        except FileNotFoundError:
            return False

    def train(self, max_steps=None):
        device = next(self.model.parameters()).device
        if self.mixed_percision_training:
            # Creates once at the beginning of training
            self.scaler = torch.cuda.amp.GradScaler()

        while True:
            if self.is_master:
                tq = tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}')

            for features in tq if self.is_master else self.dataset:

                if max_steps is not None and self.step > max_steps:
                    self.save_to_checkpoint()
                    return

                update_params = self.substep == 0 or ((self.substep + 1) % self.grad_accum_steps == 0)

                features = _nested_map(features, lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x)

                loss = self.train_step(features, max_steps)

                if torch.isnan(loss).any():
                    raise RuntimeError(f'Detected NaN loss at step {self.step}.')
                if self.is_master:
                    tq.set_postfix(loss=f"{(loss*self.grad_accum_steps):.4f}",
                                   grad_norm=f"{self.grad_norm:.4f}")
                    if update_params:
                        if self.step % 100 == 0:
                            self._write_summary(self.step, features, loss)
                        if self.step % self.valid_loop_every == 0 and self.step > 0:
                            self.run_valid_loop()
                        if self.step > 0 and (self.step * self.grad_accum_steps) % self.ckpt_every == 0:
                            print("INFO: saving checkpoint at step {}".format(self.step))
                            self.save_to_checkpoint()

                self.substep += 1  # including grad accum steps
                if update_params:
                    self.step += 1

    ######################################################################################################

    def backward_and_update(self, loss, max_steps):
        if self.grad_accum_steps > 1:
            loss = loss / float(self.grad_accum_steps)
        if self.mixed_percision_training:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.substep == 0 or ((self.substep + 1) % self.grad_accum_steps == 0):

            if self.mixed_percision_training:
                self.scaler.unscale_(self.optimizer)
            self.grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=False)

            if self.mixed_percision_training:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        return loss

    def train_step(self, features, max_steps):
        pass

    def run_valid_loop(self):
        pass

    def _write_summary(self, step, features, loss):
        pass
