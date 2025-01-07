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
    ("max_round", "R"),
    ("stitched_method", ""),

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
    ("max_round", "R"),
    ("stitched_method", ""),
]

logbase = "logs"
base = {
    "diffusion": {
        ## model
        "model": "models.TemporalUnet",
        "diffusion": "models.GaussianDiffusion",
        "horizon": 16,
        "jump": 1,
        "jump_action": False,
        "condition": True,
        "n_diffusion_steps": 128,
        "action_weight": 10,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": False,
        "dim_mults": (1, 4, 8),
        "upsample_k": (4, 4),
        "downsample_k": (4, 4),
        "kernel_size": 5,
        "dim": 32,
        "renderer": "utils.Maze2dRenderer",
        ## dataset
        "loader": "datasets.GoalDataset",
        "termination_penalty": None,
        "normalizer": "LimitsNormalizer",
        # "preprocess_fns": ["maze2d_set_terminals"],
        "preprocess_fns": [],
        "clip_denoised": True,
        "use_padding": False,
        "max_path_length": 40000,
        ## serialization
        "logbase": logbase,
        "prefix": "stitched_diffusion_hd/",
        "exp_name": watch(diffusion_args_to_watch),
        ## training
        "n_steps_per_epoch": 10000,
        "loss_type": "l2",
        "n_train_steps": 2e6,
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
        "use_stitched_data": True,
        "use_short_data": True,
        "max_round": 7,
        "max_n_episodes": 100000,
        "stitched_method": "linear-non_straight", # "linear"

    },
    "plan": {
        "stitched_method": "linear-non_straight", # "linear"
        "batch_size": 1,
        "device": "cuda",
        ## diffusion model
        "horizon": 16,
        "jump": 1,
        "jump_action": False,
        "condition": True,
        "kernel_size": 5,
        "dim": 32,
        "n_diffusion_steps": 128,
        "normalizer": "LimitsNormalizer",
        ## serialization
        "logbase": logbase,
        "vis_freq": 10,
        # "logbase": "/common/users/cc1547/projects/diffuser/logs",
        "prefix": "plans_stitched_diffusion_hd/release",
        "exp_name": watch(plan_args_to_watch),
        "suffix": "0",
        "conditional": False,
        "transfer": "none",
        "restricted_pd": False,
        ## loading
        "diffusion_loadpath": "f:stitched_diffusion_hd/H{horizon}_T{n_diffusion_steps}_J{jump}_R{max_round}_{stitched_method}",
        "diffusion_epoch": "latest",
        "max_round": 7,
    },
}

# ------------------------ overrides ------------------------#

"""
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
"""

maze2d_xxlarge_v1 = {
    "diffusion": {
        "max_round": 10,
    },
    "plan": {
        "max_round": 10,
    },

}
