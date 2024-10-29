import torch

from params_proto.neo_proto import ParamsProto, PrefixProto, Proto


class Config(ParamsProto):
    seed = 100
    device = "cuda:0"
    prefix = "default_inv/predict_epsilon_100_1000000.0/dropout_0.25/kitchen/hl40_r3_j4"
    bucket = "/common/users/cc1547/projects/rainbow/diffstitch/diffuser/kitchen/hl"
    job_name = "predict_epsilon_100_1000000.0/dropout_0.25/kitchen_partial/hl40_r3_j4"
    dataset = "kitchen-partial-v0"
    test_ret = 0.95
    job_counter = 1

    # Stitching
    render_option = False
    render_freq = 50
    dream_len = 1
    dynamics_deviate = 1.5
    number_optimum = 20000  # 2%
    top_k = 300000  # 30%
    save_img_dir = "/root/4_20_workspace/pictures"
    dynamic_model_path = "/root/dynamic_models/kitchen_partial"
    save_data_path = "/root/autodl-tmp/open_code/augmented_data"
    dreamer_similarity = 0.80
    stitch_L = 10
    stitch_R = 40
    generate_limit = 2000000
    stitch_batch = 64
    sample_optim_batch = 512
    save_aug_freq = 5

    ## dataset
    termination_penalty = -1
    returns_scale = 2.0  # Determined using rewards from the dataset
    loader = "datasets.CondSequenceDataset"
    normalizer = "CDFNormalizer"
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True
    discount = 0.997
    max_path_length = 260
    hidden_dim = 256
    ar_inv = False
    train_only_inv = False
    data_file = None
    stitch = True
    aug_data_file = "/common/users/cc1547/dataset/rainbow/stitching_kitchen/round3_stitch_kitchen_partial_v0_H20-v1.pkl"
    jump = 4
    task_data = True
    jumps = []
    task_len = 20
    segment_return = False

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
    resume_step = 0

    # model
    model = "models.TemporalUnet"
    diffusion = "models.GaussianInvDynDiffusion"
    train_only_diffuser = True
    horizon = 37
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
    ll_loadpath = "/common/users/cc1547/projects/rainbow/diffstitch/diffuser/kitchen/hl/kitchen-partial-v0/default_inv/predict_epsilon_100_1000000.0/dropout_0.25/kitchen/ll7_r3/checkpoint"
