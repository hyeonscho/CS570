python scripts/train.py --config config.low_level --dataset "pointmaze-medium-navigate-v0"
python scripts/train.py  --config config.low_level --dataset "pointmaze-large-navigate-v0"
python scripts/train.py --config config.low_level --dataset "pointmaze-giant-navigate-v0"


python scripts/train.py --config config.hl_large_medium --dataset "pointmaze-medium-navigate-v0"
python scripts/train.py --config config.hl_large_medium --dataset "pointmaze-large-navigate-v0"
python scripts/train.py --config config.hl_giant --dataset "pointmaze-giant-navigate-v0"

python scripts/train.py --config config.diffuser_giant --dataset "pointmaze-giant-navigate-v0"