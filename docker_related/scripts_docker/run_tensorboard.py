#!/usr/bin/python3
# Using which python

import os
import argparse
import random
from datetime import datetime

# Something fast, TODO: improve it, add more features, make it an easy runner, modular, ....etc
username = os.path.expanduser('~').split('/')[-1]
to_path = "root"  # "HMBRL-KAIST"
image = username+"/diffuser_zoo"  # "hany606/hmbrl"

random.seed(datetime.now().timestamp())

parser = argparse.ArgumentParser(
    prog="run_docker.py", description="Runner", epilog="Help"
)

parser.add_argument("--device", nargs="+", type=int, default=[0], required=False)
parser.add_argument("--script", type=str, default="bash", required=False)
parser.add_argument("--bindport", type=str, default="", required=False)


args = parser.parse_args()

devices = [str(d) for d in args.device]
key = ","
# '"device=gpu_num"'
device = f"'\"device={key.join(devices)}\"'"
# device = f"'\"device={args.device}\"'"
script = args.script
text = f". $(pwd)/scripts_docker/wandb_login.bash && docker run -it {args.bindport} --rm --gpus {device} --user=$(id -u):$(id -g) -v ~/logdir:/logdir -v $(pwd):/{to_path} -v /data/{username}/datasets/d4rl:/home/{username}/.d4rl/datasets/ -v /data/datasets/d4rl:/data/datasets/d4rl --env WANDB_API_KEY=$WANDB_API_KEY --env HostName=$(hostname) --publish 6006:6006 --env ServerNum={''.join(devices)} {image} {script}"

print(text)
os.system(text)

