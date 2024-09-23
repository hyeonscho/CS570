import torch

from params_proto.neo_proto import ParamsProto, PrefixProto, Proto


class Config(ParamsProto):
    seed = 100
    device = "cuda:0"
    prefix = "diffuser/default_inv/predict_epsilon_100_1000000.0/dropout_0.25/hopper-medium-replay-v2/ll40_r2_run2"
    bucket = "/common/users/cc1547/projects/rainbow/diffstitch/diffuser/gym_mujoco/hl"
    job_name = "predict_epsilon_100_1000000.0/dropout_0.25/hopper-medium-replay-v2/ll40_r2_run2"
    dataset = "hopper-medium-replay-v2"
    test_ret = 0.25
    job_counter = 1

    # Stitching
    render_option = True
    render_freq = 50
    dream_len = 1
    dynamics_deviate = 0.56
    number_optimum = 8000  # 2%
    top_k = 120000  # 30%
    save_img_dir = "/root/4_16_workspace/pictures"
    dynamic_model_path = "/common/users/cc1547/projects/rainbow/diffstitch/dynamic/hopper-medium-replay-v2/mopo/seed_1_0826_170326-hopper_medium_replay_v2_mopo/models/ite_dynamics_model"
    save_data_path = "/root/4_16_workspace/augmented_data"
    dreamer_similarity = 0.90
    stitch_L = 10
    stitch_R = 40
    generate_limit = 2000000
    stitch_batch = 64
    sample_optim_batch = 512
    save_aug_freq = 5
    segment_return = False

    ## dataset
    termination_penalty = -100
    returns_scale = 200.0  # Determined using rewards from the dataset
    loader = "datasets.CondSequenceDataset"
    normalizer = "CDFNormalizer"
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True
    discount = 0.99
    max_path_length = 200
    hidden_dim = 256
    ar_inv = False
    train_only_inv = False
    data_file = None
    stitch = False
    task_data = True
    jump = 1
    aug_data_file = "/common/users/cc1547/dataset/rainbow/stitching_gym/round2_stitch_hopper-medium-replay-v2_H40-v2.pkl"
    data_file = None

    ## training
    n_steps_per_epoch = 10000
    loss_type = "l2"
    n_train_steps = 1e6
    batch_size = 32
    learning_rate = 2e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 1000
    save_freq = 100000
    sample_freq = 10000
    n_saves = 5
    save_parallel = False
    n_reference = 8
    save_checkpoints = True
    n_samples = 10

    # model
    model = "models.TemporalUnet"
    diffusion = "models.GaussianInvDynDiffusion"
    train_only_diffuser = False
    horizon = 40
    n_diffusion_steps = 100
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True
    dim_mults = (1, 4, 8)
    returns_condition = True
    calc_energy = False
    dim = 128
    condition_dropout = 0.25
    condition_guidance_w = 1.2
    renderer = "utils.MuJoCoRenderer"
