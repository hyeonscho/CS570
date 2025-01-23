#!/bin/bash

env_names=('medium' 'large' 'giant' 'teleport')
# datasets=('pointmaze-medium-navigate-v0' 'pointmaze-large-navigate-v0' 'pointmaze-giant-navigate-v0' 'pointmaze-teleport-navigate-v0')

for env_name in "${env_names[@]}"; do
    echo "Running for environment: $env_name"
    python scripts/plan_maze2d_nav_replanning.py --config config.og_navigate_$env_name --dataset "pointmaze-$env_name-navigate-v0"
done

echo "All tasks completed."