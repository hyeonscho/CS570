# short_seq_len < max_jump
import socket

from diffuser.utils import watch

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ("prefix", ""),
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ("short_seq_len", "S"),
    ("jumps", "J"),
    ("action_weight", "AW"),
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
    ("short_seq_len", "S"),
    # ("jumps", "J"),
    ("restricted_pd", "rpd"),
    ("max_round", "R"),
    ("stitched_method", ""),
]

logbase = "logs"
base = {
    "diffusion": {
        ## model
        "model": "models.TemporalUnet",
        "diffusion": "models.GaussianDiffusionHMDNoLevelWeight",
        "horizon": 400,
        # "jump": 15,
        "jump_action": "none",
        "condition": True,
        "n_diffusion_steps": 256,
        "action_weight": 10,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": False,
        "dim_mults": (1, 4, 8),
        "upsample_k": (3, 3, 3),
        "downsample_k": (3, 3, 3),
        "kernel_size": 5,
        "dim": 32,
        "renderer": "utils.Maze2dRenderer",
        ## dataset
        "loader": "datasets.GoalDatasetHMDMultiscale2",
        "termination_penalty": None,
        "normalizer": "LimitsNormalizer",
        # "preprocess_fns": ["maze2d_set_terminals"],
        "preprocess_fns": [],
        "clip_denoised": True,
        "use_padding": False,
        "max_path_length": 40000,
        ## serialization
        "logbase": logbase,
        "prefix": "stitched_hmd_multiscale2/",
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
        
        "jumps": [1, 1, 1, 10, 15, 20],
        "short_seq_len": 21,
        "level_dim": None,
        "use_stitched_data": True,
        "use_short_data": True,
        "max_round": 3,
        "max_n_episodes": 100000,
        "stitched_method": "linear-non_overlap", # "linear"


    },
    "plan": {
        "stitched_method": "linear-non_overlap", # "linear"
        "action_weight": 10,
        "max_round": 3,
        "batch_size": 1,
        "device": "cuda",
        ## diffusion model
        "horizon": 400,
        # "jump": 15,
        "jumps": [1, 1, 1, 10, 15, 20],
        "short_seq_len": 21,
        "level_dim": None,
        "jump_action": "none",
        "attention": False,
        "condition": True,
        "kernel_size": 5,
        "dim": 32,
        "mask": False,
        "n_diffusion_steps": 256,
        "normalizer": "LimitsNormalizer",
        "logbase": logbase,
        ## serialization
        "vis_freq": 10,
        "prefix": "plans_stitched_hmd_multiscale2/",
        "exp_name": watch(plan_args_to_watch),
        "suffix": "0",
        "conditional": False,
        "transfer": "none",
        "restricted_pd": False,
        ## loading
        "diffusion_loadpath": "f:stitched_hmd_multiscale2/H{horizon}_T{n_diffusion_steps}_S{short_seq_len}_J{jumps}_AW{action_weight}_R{max_round}_{stitched_method}",
        "diffusion_epoch": "latest",

        "classifier_loadpath": "f:stitched_hmd_classifier/H{horizon}_T{n_diffusion_steps}_S{short_seq_len}_J{jumps}_AW{action_weight}_R{max_round}_{stitched_method}",
        # "classifier_loadpath": "f:diffusion_hmd_classifier/H{horizon}_T{n_diffusion_steps}_S{short_seq_len}_J{jumps}",
        "classifier_epoch": "latest"#"latest", #400000#

    },
}

# ------------------------ overrides ------------------------#

"""
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
"""

maze2d_umaze_v1 = {
    "diffusion": {
        "horizon": 120,
        "n_diffusion_steps": 64,
        "upsample_k": (4, 4, 4),
        "downsample_k": (4, 4, 4),
    },
    "plan": {
        "horizon": 120,
        "n_diffusion_steps": 64,
    },
}

maze2d_large_v1 = {
    "diffusion": {
        "horizon": 400,
        "n_diffusion_steps": 256,
        "upsample_k": (3, 3),
        "downsample_k": (3, 3),
        
        # original of HD
        # "upsample_k": (3, 3, 4),
        # "downsample_k": (4, 3, 3),

        # "upsample_k": (4, 4, 4),
        # "downsample_k": (4, 3, 3),
    },
    "plan": {
        "horizon": 400,
        "n_diffusion_steps": 256,
    },
}

maze2d_giant_v1 = {
    "diffusion": {
        "horizon": 500,
        "n_diffusion_steps": 256,
        "upsample_k": (3, 4),
        "downsample_k": (4, 3),
        "jumps": [1, 1, 1, 1, 6, 10, 15, 20],
        "short_seq_len": 26,
    },
    "plan": {
        "jumps": [1, 1, 1, 1, 6, 10, 15, 20],
        "short_seq_len": 26,
        "horizon": 500,
        "n_diffusion_steps": 256,
    },
}


maze2d_xxlarge_v1 = {
    "diffusion": {
        "max_round": 3,
        "horizon": 784,
        "n_diffusion_steps": 256,
        "upsample_k": (3, 3),
        "downsample_k": (3, 3),
        "jumps": [1, 1, 1, 1, 1, 1, 6, 10, 15, 20, 24, 28],
        "short_seq_len": 29, # 780 / 28 + 1 = 28 + 1 = 29
    },
    "plan": {
        "max_round": 3,
        "jumps": [1, 1, 1, 1, 1, 1, 6, 10, 15, 20, 24, 28],
        "short_seq_len": 29, # 780 / 28 + 1 = 28 + 1 = 29
        "horizon": 784,
        "n_diffusion_steps": 256,
    },
}
