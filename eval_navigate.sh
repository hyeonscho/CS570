#!/bin/bash

# python scripts/plan_maze2d_valueGuidance.py --config config.og_navigate_giant_valueGuidance_1000 --dataset "pointmaze-giant-navigate-v0"
# python scripts/plan_maze2d_valueGuidance.py --config config.og_navigate_large_valueGuidance_500 --dataset "pointmaze-large-navigate-v0"
# python scripts/plan_maze2d_valueGuidance.py --config config.og_navigate_medium_valueGuidance_500 --dataset "pointmaze-medium-navigate-v0"
# python scripts/plan_maze2d_nav_replanning_valueGuidance.py --config config.og_navigate_medium_valueGuidance_500 --dataset "pointmaze-medium-navigate-v0"
# python scripts/plan_maze2d_nav_replanning_valueGuidance.py --config config.og_navigate_large_valueGuidance_500 --dataset "pointmaze-large-navigate-v0"
# python scripts/plan_maze2d_nav_replanning_valueGuidance.py --config config.og_navigate_giant_valueGuidance_1000 --dataset "pointmaze-giant-navigate-v0"
# python scripts/plan_maze2d_antmaze_valueGuidance.py --config config.og_antmaze_navigate_medium_valueGuidance_500 --dataset "antmaze-medium-navigate-v0"
# python scripts/plan_maze2d_antmaze_valueGuidance.py --config config.og_antmaze_navigate_large_valueGuidance_500 --dataset "antmaze-large-navigate-v0"
# python scripts/plan_maze2d_antmaze_valueGuidance.py --config config.og_antmaze_navigate_giant_valueGuidance_1000 --dataset "antmaze-giant-navigate-v0"
# python scripts/plan_maze2d_antmaze_replanning_valueGuidance.py --config config.og_antmaze_navigate_medium_valueGuidance_500 --dataset "antmaze-medium-navigate-v0"
# python scripts/plan_maze2d_antmaze_replanning_valueGuidance.py --config config.og_antmaze_navigate_large_valueGuidance_500 --dataset "antmaze-large-navigate-v0"
# python scripts/plan_maze2d_antmaze_replanning_valueGuidance.py --config config.og_antmaze_navigate_giant_valueGuidance_1000 --dataset "antmaze-giant-navigate-v0"

# echo "All tasks completed."
python scripts/eval_diffuser.py --config config.diffuser_medium --dataset "pointmaze-medium-navigate-v0"
python scripts/eval_diffuser.py --config config.diffuser_large --dataset "pointmaze-large-navigate-v0"


python scripts/eval_hd.py  --config config.high_medium --dataset "pointmaze-medium-navigate-v0"
python scripts/eval_hd.py  --config config.high_large --dataset "pointmaze-large-navigate-v0"