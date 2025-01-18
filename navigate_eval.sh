#!/bin/bash

datasets=('pointmaze-medium-navigate-v0' 'pointmaze-large-navigate-v0' 'pointmaze-giant-navigate-v0' 'pointmaze-teleport-navigate-v0')

for dataset in "${datasets[@]}"; do
    echo "Running for dataset: $dataset"
    python scripts/plan_maze2d.py --config config.og_navigate --dataset "$dataset"
done

echo "All tasks completed."