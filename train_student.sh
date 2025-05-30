python scripts/train.py --config config.low_medium_distill --dataset "pointmaze-medium-navigate-v0"
python scripts/train.py --config config.high_medium_distill --dataset "pointmaze-medium-navigate-v0"
# python scripts/train.py  --config config.low_level --dataset "pointmaze-large-navigate-v0"
# python scripts/train.py --config config.low_level --dataset "pointmaze-giant-navigate-v0"


# python scripts/train.py --config config.hl_large_medium --dataset "pointmaze-medium-navigate-v0"
# python scripts/train.py --config config.hl_large_medium --dataset "pointmaze-large-navigate-v0"
# python scripts/train.py --config config.hl_giant --dataset "pointmaze-giant-navigate-v0"

# python scripts/train.py --config config.diffuser_medium_distill --dataset "pointmaze-medium-navigate-v0"
# python scripts/train.py --config config.diffuser_large_distill --dataset "pointmaze-large-navigate-v0"
python scripts/train.py --config config.diffuser_giant_distill --dataset "pointmaze-giant-navigate-v0"

python scripts/train.py --config config.low_large_distill --dataset "pointmaze-large-navigate-v0"
python scripts/train.py --config config.high_large_distill --dataset "pointmaze-large-navigate-v0"
