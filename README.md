# Code base for DiffStitch:

Our code is built on the codebase of DiffStitch: https://github.com/guangheli12/DiffStitch.git.


## Environment set up:
See `environment.yml`.

## Method
We mainly have two stage: stitching and long-horizon sequence modeling with HD.

1. Stitching:
   - Train a low-level diffuser and a dynamic model
   - Stitching long trajectories
2. Long-horizon sequence modeling:

## Training
### Train the low-level Diffuser
```
cd scripts
python train.py --config walker2d_medium_v2_task
```

### Train Dynamic models 
```
    cd scripts/mopo 
    python train.py --task walker2d-medium-v2
```

### Evaluate the low-level Diffuser to search for a good `test_ret`, which will be used during stitching.
```
    cd scripts 
    python diffstitch_gym_mujoco_eval.py --task walker2d_medium_v2_task
```

### Stitch to generate long trajectories:
Currently the stitching is conducted sequentially, which means if we want to stitch 3 short trajectories together, we need stitch 2 first, and then stitch the resulted mediated long trajectories with a third short trajectorie, and so on.
```
    python diffstitch_gym_mujoco_stitch_v3.py
    python diffstitch_gym_mujoco_stitch_round2.py
```
### Post-process the stitched long tracjectories:
See jupyter notebook `diffstitch_gym_post_process-walker2d-me_invest.ipynb`. We need to make the dataset format aligned with the d4rl dataset format. Also see the episoidc return of the stitched data, to set the `reward_scale` value.

### Train level-conditionig planner:
```
cd scripts
python train.py --config walker2d_medium_v2_cl
```

## Evaluation
```
cd scripts
python cl_diffstitch_gym_eval_r1.py
```
It also searches for a good conditioning testing return.


## Others
Understanding the statistics of the dataset is benefitial for choosing the hyper-parameters, especially the `return_scale`, for training.
The jupyter notebook `DMC_data_investigate` is what I used for the DMC environment.


### ERROR & BUGS
- from tap import Tap  
    Install this package 
    ```python 
    pip install typed-argument-parser
    ```
