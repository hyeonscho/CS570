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
# Previous models used only the 10k last episodes -> I do not want to mess up with their training
progressive_distillation = args.progressive_distillation if hasattr(args, "progressive_distillation") else False
teacher_path = args.teacher_path if hasattr(args, "teacher_path") else None

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
if progressive_distillation:
    assert teacher_path is not None, "Teacher path must be provided for progressive distillation."
    import copy, torch
    # teacher_model = copy.deepcopy(model).to(args.device)
    # state_dict = torch.load(teacher_path, map_location=args.device)
    # if "model" in state_dict:
    #     state_dict = state_dict["model"]  # Extract model parameters if nested
    # filtered_state_dict = {k: v for k, v in state_dict.items() if k in teacher_model.state_dict()}
    # teacher_model.load_state_dict(filtered_state_dict, strict=False)
    # teacher_model.eval()
    # teacher_model.n_timesteps = args.n_diffusion_steps * 4
    teacher_model = utils.load_diffusion(args.logbase, args.dataset, args.teacher_path, epoch='latest').ema
else:
    teacher_model = None
diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)


# -----------------------------------------------------------------------------#
# ------------------------ test forward & backward pass -----------------------#
# -----------------------------------------------------------------------------#

utils.report_parameters(model)

print("Testing forward...", end=" ", flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch, teacher_model=teacher_model)
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
    trainer.train(n_train_steps=args.n_steps_per_epoch, writer=train_writer, teacher_model=teacher_model)
