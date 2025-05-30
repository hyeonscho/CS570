#!/usr/bin/python3
# Using which python

# Something fast, TODO: improve it, add more features, make it an easy runner, modular, ....etc

import os
import argparse
import random
from datetime import datetime

random.seed(datetime.now().timestamp())

parser = argparse.ArgumentParser(
    prog="run_pinpad.py", description="Runner", epilog="Help"
)

parser.add_argument("--device", nargs="+", type=int, default=[0], required=False)

args = parser.parse_args()

devices = [str(d) for d in args.device]

key = " "

device = f"{key.join(devices)}"

# script = f"'\"jupyter-lab --ip 0.0.0.0 --no-browser --allow-root\"'"
script = f"'\"./scripts_docker/_run_juypter.sh\"'"
text = f"python $(pwd)/scripts_docker/run_docker.py --device {device} --script {script} --bindport '-p 8888:8888' "

print(text)
os.system(text)
