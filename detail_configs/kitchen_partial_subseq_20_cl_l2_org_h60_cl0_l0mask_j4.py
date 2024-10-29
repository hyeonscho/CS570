import torch

from params_proto.neo_proto import ParamsProto, PrefixProto, Proto


class Config(ParamsProto):
    seed = 100
    device = "cuda:0"
    prefix = "default_inv/predict_epsilon_100_1000000.0/dropout_0.25/kitchen_partial/subseq_20_l2_org_h60_cl0_l0mask_j4"
    bucket = "/common/users/cc1547/projects/rainbow/diffstitch/diffuser/kitchen/cl"
    job_name = "predict_epsilon_100_1000000.0/dropout_0.25/kitchen_partial/subseq_20_l2_org_h60_cl0_l0mask_j4"
    dataset = "kitchen-partial-v0"
    test_ret = 0.9
    job_counter = 1

    # Stitching
    render_option = False
    render_freq = 50
    dream_len = 1
    dynamics_deviate = 0.2
    number_optimum = 20000  # 2%
    top_k = 300000  # 30%
    save_img_dir = "/root/4_20_workspace/pictures"
    dynamic_model_path = "/common/users/cc1547/projects/rainbow/diffstitch/dynamic/kitchen-partial-v0/mopo/seed_1_lr0.001_0923_135030-kitchen_partial_v0_mopo/models/ite_dynamics_model"
    save_data_path = "/root/autodl-tmp/open_code/augmented_data"
    dreamer_similarity = 0.99
    stitch_L = 10
    stitch_R = 40
    generate_limit = 2000000
    stitch_batch = 64
    sample_optim_batch = 512
    save_aug_freq = 5
    std_coef = 4.0

    ## dataset
    termination_penalty = -1
    returns_scale = 2.0  # Determined using rewards from the dataset
    loader = "datasets.CondCLSequenceDatasetV2"
    normalizer = "CDFNormalizer"
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True
    discount = 0.997
    max_path_length = 280
    hidden_dim = 256
    ar_inv = False
    train_only_inv = False
    data_file = None
    stitch = False
    task_data = False
    jump = 1
    jumps = [1, 4]
    segment_return = False
    aug_data_file = None
    task_len = 20

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
    diffusion = "models.GaussianInvDynDiffusionCL"
    horizon = 61
    train_only_diffuser = False
    n_diffusion_steps = 100
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True
    level_epsilon = True
    dim_mults = (1, 4, 8)
    returns_condition = True
    level_condition = False
    calc_energy = False
    dim = 128
    condition_dropout = 0.25
    condition_guidance_w = 1.2
    renderer = "utils.MuJoCoRenderer"
    level_dim = 0
