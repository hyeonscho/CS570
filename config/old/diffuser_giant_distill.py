import socket

from diffuser.utils import watch

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ("prefix", "diffuser"),
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
progressive_distillation = True
teacher_path = 'diffuser_ogbench_navigate_valueGuidance/H1000_T256_J1/'
# teacher_path = './logs/pointmaze-giant-navigate-v0/diffuser_ogbench_navigate_valueGuidance/H1000_T128_J1/state_20000.pt'
# teacher_path = './logs/pointmaze-giant-navigate-v0/diffuser_ogbench_navigate_distill/H1000_T64_J1/state_18800.pt'
logbase = "logs"
base = {
    "diffusion": {
        "progressive_distillation": progressive_distillation,
        "teacher_path": teacher_path,
        ## model
        "model": "models.TemporalUnet",
        "diffusion": "models.GaussianDiffusion",
        "horizon": 1000, # 1000
        "jump": 1,
        "jump_action": 1,#"none",
        "condition": True,
        "n_diffusion_steps": 64,
        "action_weight": 0.0,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": True,
        "dim_mults": (1, 4, 8),
        "upsample_k": (4, 4, 3),
        "downsample_k": (3, 3, 3),
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
        "max_path_length": 1000,
        ## serialization
        "logbase": logbase,
        "prefix": "distillation_after_fix/",
        "exp_name": watch(diffusion_args_to_watch),
        ## training
        "n_steps_per_epoch": 10000,
        "loss_type": "l2",
        "n_train_steps": 3e4, # 2e6
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
        "horizon": 1000,
        "jump": 1,
        "jump_action": 1, #"none",
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
        "restricted_pd": True,
        ## loading
        "diffusion_loadpath": "f:diffuser_ogbench_navigate_valueGuidance/H{horizon}_T{n_diffusion_steps}_J{jump}",
        "diffusion_epoch": "latest", #1000000,
    },
}

# ------------------------ overrides ------------------------#

# """
#     maze2d maze episode steps:
#         umaze: 150
#         medium: 250
#         large: 600
# """

# maze2d_umaze_v1 = {
#     "diffusion": {
#         "horizon": 120,
#         "n_diffusion_steps": 64,
#         "upsample_k": (4, 4, 4),
#         "downsample_k": (4, 4, 4),
#     },
#     "plan": {
#         "horizon": 120,
#         "n_diffusion_steps": 64,
#     },
# }

# maze2d_large_v1 = {
#     "diffusion": {
#         "horizon": 300,
#         "n_diffusion_steps": 256,
#         "upsample_k": (4, 4),
#         "downsample_k": (3, 3),
#     },
#     "plan": {
#         "horizon": 300,
#         "n_diffusion_steps": 256,
#     },
# }
