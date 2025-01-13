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
    ("jumps", "J")
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
        "loader": "datasets.GoalDatasetHMDMultiscale",
        "termination_penalty": None,
        "normalizer": "LimitsNormalizer",
        "preprocess_fns": ["maze2d_set_terminals"],
        "clip_denoised": True,
        "use_padding": False,
        "max_path_length": 40000,
        ## serialization
        "logbase": logbase,
        "prefix": "diffusion_hmd_classifier/",
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

        "classifier": "models.LevelClassifier",
        "num_layers": 3,
        "hidden_dim": 256,

    },
    "plan": {
        "batch_size": 1,
        "device": "cuda",
        ## diffusion model
        "horizon": 400,
        # "jump": 15,
        "jumps": [1, 1, 1, 10, 15, 20],
        "short_seq_len": 11,
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
        "prefix": "plans/release",
        "exp_name": watch(plan_args_to_watch),
        "suffix": "0",
        "conditional": False,
        "transfer": "none",
        "restricted_pd": False,
        ## loading
        "diffusion_loadpath": "f:diffusion_hmd/H{horizon}_T{n_diffusion_steps}_J{jump}",
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
        "upsample_k": (4, 3),
        "downsample_k": (3, 3),
        # "upsample_k": (4, 4, 4),
        # "downsample_k": (4, 3, 3),
    },
    "plan": {
        "horizon": 400,
        "n_diffusion_steps": 256,
    },
}

maze2d_xxlarge_v2 = {
    "diffusion": {
        "max_path_length": 300000,
        "horizon": 784,
        "n_diffusion_steps": 256,
        "upsample_k": (3, 3),
        "downsample_k": (3, 3),
        "jumps": [1, 1, 1, 1, 1, 1, 6, 10, 15, 20, 24, 28],
        "short_seq_len": 29, # 780 / 28 + 1 = 28 + 1 = 29
    },
    "plan": {
        "max_path_length": 300000,
        "jumps": [1, 1, 1, 1, 1, 1, 6, 10, 15, 20, 24, 28],
        "short_seq_len": 29, # 780 / 28 + 1 = 28 + 1 = 29
        "horizon": 784,
        "n_diffusion_steps": 256,
    },
}


maze2d_xxlargec_v1 = {
    "diffusion": {
        "max_path_length": 300000,
        "horizon": 784,
        "n_diffusion_steps": 256,
        "upsample_k": (3, 3),
        "downsample_k": (3, 3),
        "jumps": [1, 1, 1, 1, 1, 1, 6, 10, 15, 20, 24, 28],
        "short_seq_len": 29, # 780 / 28 + 1 = 28 + 1 = 29
    },
    "plan": {
        "max_path_length": 300000,
        "jumps": [1, 1, 1, 1, 1, 1, 6, 10, 15, 20, 24, 28],
        "short_seq_len": 29, # 780 / 28 + 1 = 28 + 1 = 29
        "horizon": 784,
        "n_diffusion_steps": 256,
    },
}
