import socket

from diffuser.utils import watch

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ("prefix", ""),
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ("jump", "J"),
]

plan_args_to_watch = [
    ("prefix", ""),
    ##
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ("value_horizon", "V"),
    ("discount", "d"),
    ("normalizer", ""),
    ("batch_size", "b"),
    ##
    ("conditional", "cond"),
    ("jump", "J"),
    ("restricted_pd", "rpd"),
]

logbase = "logs"
prefixGlobal = "_low_level_distill/"
progressive_distillation = True
# teacher_path = './logs/pointmaze-medium-navigate-v0/diffusion/H16_T128_J1/state_120000.pt'
teacher_path = "./logs/pointmaze-large-navigate-v0/diffusion/H16_T128_J1/state_196000.pt"
base = {
    "diffusion": {
        "progressive_distillation": progressive_distillation,
        "teacher_path": teacher_path,
        ## model
        "model": "models.TemporalUnet",
        "diffusion": "models.GaussianDiffusion",
        "horizon": 16,
        "jump": 1,
        "jump_action": False,
        "condition": True,
        "n_diffusion_steps": 64, # 128 => 32 => 8 => 2
        "action_weight": 0.0,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": True,
        "dim_mults": (1, 4, 8),
        "upsample_k": (4, 4),
        "downsample_k": (4, 4),
        "kernel_size": 5,
        "dim": 32,
        "renderer": "utils.Maze2dRenderer",
        ## dataset
        "loader": "datasets.OGMaze2dOfflineRLDataset",
        "only_start_condition": False, ########## Value Guidance
        "termination_penalty": None,
        "normalizer": "LimitsNormalizer",
        "preprocess_fns": ["maze2d_set_terminals"],
        "clip_denoised": True,
        "use_padding": False,
        "max_path_length": 990,  # 1000
        ## serialization
        "logbase": logbase,
        "prefix": "diffusion_distill/",
        "exp_name": watch(diffusion_args_to_watch),
        ## training
        "n_steps_per_epoch": 10000,
        "loss_type": "l2",
        "n_train_steps": 12e4,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 2,
        "ema_decay": 0.995,
        "save_freq": 1000,
        "sample_freq": 10000,
        "n_saves": 50,
        "save_parallel": False,
        "n_reference": 50,
        "n_samples": 10,
        "bucket": None,
        "device": "cuda",
    },
    "plan": {
        "batch_size": 1,
        "device": "cuda",
        ## diffusion model
        "horizon": 16,
        "jump": 1,
        "jump_action": 1, #"none",
        "attention": False,
        "condition": True,
        "kernel_size": 5,
        "dim": 32,
        "mask": False,
        "n_diffusion_steps": 32,
        "normalizer": "LimitsNormalizer",
        "logbase": logbase,
        ## serialization
        "vis_freq": 10,        
        "prefix": "plans/release",
        "prefixGlobal": prefixGlobal,
        "exp_name": watch(plan_args_to_watch),
        "suffix": "0",
        "conditional": False,
        "transfer": "none",
        "restricted_pd": True,
        ## loading
        "diffusion_loadpath": "f:diffusion/H{horizon}_T{n_diffusion_steps}_J{jump}",
        "diffusion_epoch": "latest",
    },
}

# ------------------------ overrides ------------------------#

"""
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
"""