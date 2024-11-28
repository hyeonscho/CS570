import os
import copy
import numpy as np
import torch
import einops
import pdb
import diffuser
from copy import deepcopy
from diffuser.datasets.sequence import SequenceDataset
from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
from ml_logger import logger
from torch.utils.data import random_split

from diffuser.utils.debug import debug
from .training import EMA
import math
from typing import List

def cycle(dl):
    while True:
        for data in dl:
            yield data

# https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#random_split
# The working version of torch (it is implemented differently than the one in the link)
def generate_lengths(dataset, lengths):
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
    return lengths

class TrainerClassifier(object):
    def __init__(
        self,
        model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=1000,
        sample_freq=1000,
        save_freq=1000,
        label_freq=50000,
        save_parallel=False,
        results_folder="./results",
        n_reference=8,
        n_samples=2,
        bucket=None,
        train_device="cuda",
        save_checkpoints=False,
        ):
        super().__init__()
        self.model = model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.save_checkpoints = save_checkpoints
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
                
        self.train_dataset, self.eval_dataset = random_split(dataset, generate_lengths(dataset, [0.9, 0.1]))
        # self.train_dataset = dataset
        # self.eval_dataset = dataset
        
        self.dataloader = cycle(
            torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=train_batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
            )
        )
        
        self.eval_dataloader = cycle(
            torch.utils.data.DataLoader(
                self.eval_dataset,
                batch_size=train_batch_size*4,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
            )
        )
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

        self.device = train_device


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())
        
    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

        
    def train(self, n_train_steps, writer=None):
        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                # batch: trajectories, conditions, returns, levels
                # batch.trajectories: BxTxD
                # batch.conditions: BxD
                
                observations = batch.trajectories
                # Take the first and the last
                observations = torch.cat([observations[:, 0, None], observations[:, -1, None]], dim=1)
                labels = batch.levels.argmax(dim=-1).long()
                                
                loss, infos = self.model.loss(observations, labels)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            # if self.step % self.update_ema_every == 0:
            #     self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)
                self.eval(self.eval_dataloader, writer, legend="eval")
                self.eval(self.dataloader, writer, legend="train")


            if self.step % self.log_freq == 0:
                infos_str = " | ".join(
                    [f"{key}: {val:8.4f}" for key, val in infos.items()]
                )
                batch_time = timer()
                print(f"{self.step}: {loss:8.4f} | {infos_str} | t: {batch_time:8.4f}")
                writer.add_scalar(
                    "total_loss", loss.detach().item(), global_step=self.step
                )
                writer.add_scalar("batch_time", batch_time, global_step=self.step)
                writer.add_scalars("infos", infos, global_step=self.step)
                writer.flush()

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            self.step += 1

    def eval(self, dataloader, writer, legend):
        timer = Timer()
        with torch.no_grad():
            self.model.eval()
            batch = next(dataloader)
            batch = batch_to_device(batch, device=self.device)
            # batch: trajectories, conditions, returns, levels
            # batch.trajectories: BxTxD
            # batch.conditions: BxD
            
            observations = batch.trajectories
            # Take the first and the last
            observations = torch.cat([observations[:, 0, None], observations[:, -1, None]], dim=1)
            labels = batch.levels.argmax(dim=-1).long()
            
            loss, infos = self.model.loss(observations, labels)
            
            
            preds = self.model(observations).softmax(-1).argmax(-1)

            val_acc = torch.sum(preds == labels)/len(labels)
            # print(preds)
            # print(labels)
            # print(preds == labels)

            
            infos[f"{legend}_acc"] = val_acc
            infos["len_labels"] = len(labels)
            

            self.optimizer.zero_grad()

            infos_str = " | ".join(
                [f"{key}: {val:8.4f}" for key, val in infos.items()]
            )
            batch_time = timer()
            print(f"{self.step}: {loss:8.4f} | {infos_str} | t: {batch_time:8.4f}")
            writer.add_scalar(
                f"total_loss_{legend}", loss.detach().item(), global_step=self.step
            )
            writer.add_scalar("batch_time", batch_time, global_step=self.step)
            writer.add_scalars("infos", infos, global_step=self.step)
            writer.flush()

            self.model.train()
        

    def save(self, epoch, prefix=None):
        """
        saves model and ema to disk;
        syncs to storage bucket if a bucket is specified
        """
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.model.state_dict(), #self.ema_model.state_dict(),
        }
        if prefix is not None:
            savepath = os.path.join(self.logdir, f"{prefix}_state.pt")
        else:
            savepath = os.path.join(self.logdir, f"state_{epoch}.pt")
        torch.save(data, savepath)
        print(f"[ utils/training ] Saved model to {savepath}")
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)


    def load(self, epoch):
        """
        loads model and ema from disk
        """
        loadpath = os.path.join(self.logdir, f"state_{epoch}.pt")
        data = torch.load(loadpath)

        self.step = data["step"]
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])
        print(f"trainner load from {loadpath}, step {self.step}")


    def render_reference(self, batch_size=10):
        """
        renders training points
        """

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
            )
        )
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:, None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.model.action_dim :]
        observations = self.dataset.normalizer.unnormalize(
            normed_observations, "observations"
        )

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join(self.logdir, f"_sample-reference.png")
        self.renderer.composite(savepath, observations)

