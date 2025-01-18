#!/usr/bin/python3
# Using which python

import os
import argparse
import random
from datetime import datetime

# Something fast, TODO: improve it, add more features, make it an easy runner, modular, ....etc
# username = os.path.expanduser('~').split('/')[-1]
username = "doojin"
to_path = "root"  # "HMBRL-KAIST"
image = username+"/diffuser"  # "hany606/hmbrl"

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
text = f". $(pwd)/docker_related/scripts_docker/wandb_login.bash && docker run --name dev -it -d {args.bindport} --rm --gpus {device} --user=$(id -u):$(id -g) -v ~/logdir:/logdir -v $(pwd):/{to_path} -v /home/baek1127/ogbench:/home/baek1127/ogbench/ -v /data/baek1127/datasets/d4rl:/home/baek1127/.d4rl/datasets/ -v /home/baek1127/.ogbench/data/:/home/baek1127/.ogbench/data/ --env WANDB_API_KEY=$WANDB_API_KEY --env HostName=$(hostname) --env ServerNum={''.join(devices)} {image} {script}"

print(text)
os.system(text)
