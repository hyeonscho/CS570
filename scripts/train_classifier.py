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
    config: str = "config.maze2d_hl"


args = Parser().parse_args("diffusion")

# -----------------------------------------------------------------------------#
# ---------------------------------- dataset ----------------------------------#
# -----------------------------------------------------------------------------#
use_stitched_data = args.use_stitched_data if hasattr(args, "use_stitched_data") else False
use_short_data = args.use_short_data if hasattr(args, "use_short_data") else False
max_round = args.max_round if hasattr(args, "max_round") else -1
# Previous models used only the 10k last episodes -> I do not want to mess up with their training
max_n_episodes = args.max_n_episodes if hasattr(args, "max_n_episodes") else 10000
stitched_method = args.stitched_method if hasattr(args, "stitched_method") else "linear" # to enable backward compatibility

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
    use_stitched_data=use_stitched_data,
    use_short_data=use_short_data,
    max_round=max_round,
    max_n_episodes=max_n_episodes,
    stitched_method=stitched_method,
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
action_dim = dataset.action_dim #* args.jump
level_dim = len(args.jumps) if args.level_dim is None else args.level_dim
if args.jump_action == "none":
    action_dim = 0




# -----------------------------------------------------------------------------#
# ------------------------------ model & trainer ------------------------------#
# -----------------------------------------------------------------------------#

classifier_config = utils.Config(
    args.classifier,
    savepath=(args.savepath, "classifier_config.pkl"),
    observation_dim=observation_dim,
    action_dim=action_dim,
    # loss_type=Config.loss_type,
    device=args.device,
    num_classes=level_dim, # changed to regressor. previously: num_classes=dataset.leveldim
    hidden_dim=args.hidden_dim,
    num_layers=args.num_layers,
)
trainer_config = utils.Config(
    utils.TrainerClassifier,
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

model = classifier_config()

trainer = trainer_config(model, dataset, renderer)


# -----------------------------------------------------------------------------#
# ------------------------ test forward & backward pass -----------------------#
# -----------------------------------------------------------------------------#

utils.report_parameters(model)

# print("Testing forward...", end=" ", flush=True)
# batch = utils.batchify(dataset[0])
# from diffuser.utils.debug import debug
# debug()
# loss, _ = model.loss(*batch)
# loss.backward()
# print("✓")


# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

train_writer = SummaryWriter(log_dir=args.savepath + "-train")
for i in range(n_epochs):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}")

    trainer.train(n_train_steps=args.n_steps_per_epoch, writer=train_writer)
