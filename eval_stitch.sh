#!/bin/bash

datasets=('pointmaze-medium-stitch-v0' 'pointmaze-large-stitch-v0' 'pointmaze-giant-stitch-v0' 'pointmaze-teleport-stitch-v0')

for dataset in "${datasets[@]}"; do
    echo "Running for dataset: $dataset"
    python scripts/plan_maze2d_stitch.py --config config.og_stitch --dataset "$dataset"
done

echo "All tasks completed."