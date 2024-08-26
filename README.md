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
python train.py --config kitchen_partial_task
```

### Train Dynamic models 
```
    cd scripts/mopo 
    python train.py --task kitchen-partial-v0
```

### Stitch to generate long trajectories:
Currently the stitching is conducted sequentially, which means if we want to stitch 3 short trajectories together, we need stitch 2 first, and then stitch the resulted mediated long trajectories with a third short trajectorie, and so on.
```
    python diffstitch_kitchen_clean.py
    python diffstitch_kitchen_clean_round_later.py
    python diffstitch_kitchen_clean_round_later_r3.py
```
### Post-process the stitched long tracjectories:
See jupyter notebook `diffstitch_kitchen_post_process.ipynb`. We need to make the dataset format aligned with the d4rl dataset format.

### Train high-level planner:
```
cd scripts
python train.py --config kitchen_partial_hl_r3
```

## Evaluation
```
cd scripts
python hl_diffstitch_kitchen_eval.py
```
It also searches for a good conditioning testing return.
To evaluate the flat stitching method:
```
cd scripts
python diffstitch_kitchen_eval.py
```

## Others
Understanding the statistics of the dataset is benefitial for choosing the hyper-parameters, especially the `return_scale`, for training.
The jupyter notebook `kitchen_data_investigate` is what I used for the Kitchen environment.


### ERROR & BUGS
- from tap import Tap  
    Install this package 
    ```python 
    pip install typed-argument-parser
    ```