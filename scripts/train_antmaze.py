import os
import sys

os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_RENDERER"] = "egl"


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diffuser.utils as utils
from torch.utils.tensorboard import SummaryWriter


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class Parser(utils.Parser):
    dataset: str = None
    config: str = None


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
    jump=args.jump,
    jump_action=args.jump_action,
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
action_dim = dataset.action_dim * args.jump
if args.jump_action == "none":
    action_dim = 0


# -----------------------------------------------------------------------------#
# ------------------------------ model & trainer ------------------------------#
# -----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, "model_config.pkl"),
    horizon=args.horizon // args.jump,
    transition_dim=observation_dim + action_dim,
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
    horizon=args.horizon // args.jump,
    condition=args.condition,
    observation_dim=observation_dim,
    action_dim=action_dim,
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
    utils.Trainer,
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
