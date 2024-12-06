import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from os.path import join
import pdb
from diffuser.guides.policies import Policy
import diffuser.utils as utils
import diffuser.datasets as datasets
from torch.utils.tensorboard import SummaryWriter


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class Parser(utils.Parser):
    dataset: str = "maze2d-large-v1"
    config: str = "config.maze2d_hmd"


args = Parser().parse_args("diffusion")

# -----------------------------------------------------------------------------#
# ---------------------------------- dataset ----------------------------------#
# -----------------------------------------------------------------------------#
dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    jumps=args.jumps,
    jump_action=args.jump_action,
    short_seq_len=args.short_seq_len,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, "render_config.pkl"),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
# Not gonna work for jump_action != "none" -> and Dense action setting 
level_dim = len(args.jumps) if args.level_dim is None else args.level_dim
if args.jump_action == "none":
    action_dim = 0
else:
    assert type(args.jump_action) == int # determine the numbers of actions to be included
    # assert args.jump_action <= args.jumps[-1] # make sure the action is not too big is not larger than the maximum jump
    # Actually, the current implementation works only if args.jump_action=1 due to the constraints of the low-level
    action_dim = dataset.action_dim * args.jump_action


# -----------------------------------------------------------------------------#
# ------------------------------ model & trainer ------------------------------#
# -----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, "model_config.pkl"),
    horizon=args.short_seq_len, #args.horizon // args.jump,
    transition_dim=observation_dim + action_dim + level_dim,
    cond_dim=observation_dim,
    dim=args.dim,
    dim_mults=args.dim_mults,
    kernel_size=args.kernel_size,
    device=args.device,
    upsample_k=args.upsample_k,
    downsample_k=args.downsample_k,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, "diffusion_config.pkl"),
    horizon=args.short_seq_len, #args.horizon // args.jump,
    condition=args.condition,
    observation_dim=observation_dim,
    action_dim=action_dim,
    level_dim=level_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.HMDTrainer,
    savepath=(args.savepath, "trainer_config.pkl"),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)

# -----------------------------------------------------------------------------#
# -------------------------------- instantiate --------------------------------#
# -----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)


# -----------------------------------------------------------------------------#
# ------------------------ test forward & backward pass -----------------------#
# -----------------------------------------------------------------------------#

utils.report_parameters(model)

print("Testing forward...", end=" ", flush=True)
batch = utils.batchify(dataset[0])
print(batch[0].shape)
loss, _ = diffusion.loss(*batch)
loss.backward()
print("✓")


# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
eval_sample_n = 3

train_writer = SummaryWriter(log_dir=args.savepath + "-train")
for i in range(n_epochs):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}")

    trainer.train(n_train_steps=args.n_steps_per_epoch, writer=train_writer)
