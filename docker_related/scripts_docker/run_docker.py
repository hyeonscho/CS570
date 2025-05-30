#!/usr/bin/python3
import os
import argparse
import random
from datetime import datetime

username = "hyeons"
to_path = "root"
image = "hyeons/diffuser"

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
device = f"'\"device={key.join(devices)}\"'"
script = args.script

# [수정된 부분] 컨테이너 이름을 GPU 번호 기반으로!
container_name = f"hs_gpu{devices[0]}"

# text = f""". $(pwd)/docker_related/scripts_docker/wandb_login.bash && \
# docker run --name {container_name} -it -d {args.bindport} --rm --gpus {device} \
# -v ~/logdir:/logdir -v $(pwd):/{to_path} \
# -v /home/hyeons/workspace/ogbench/:/home/hyeons/ogbench/ \
# -v /data/baek1127/datasets/d4rl:/home/hyeons/.d4rl/datasets/ \
# -v /home/hyeons/.ogbench/data/:/home/hyeons/.ogbench/data/ \
# --env WANDB_API_KEY=$WANDB_API_KEY --env HostName=$(hostname) --env ServerNum={''.join(devices)} \
# {image} {script}
# """
# --user=$(id -u):$(id -g) \
user_option = ""  # --user 없이 실행

text = f""". $(pwd)/docker_related/scripts_docker/wandb_login.bash && \
docker run --name {container_name} -it -d {args.bindport} --rm --gpus {device} \
-v ~/logdir:/logdir -v $(pwd):/{to_path} \
-v /home/hyeons/workspace/ogbench/:/home/hyeons/ogbench/ \
-v /home/hyeons/.mujoco:/home/hyeons/.mujoco \
-v /home/hyeons/.ogbench/data/:/home/hyeons/.ogbench/data/ \
{user_option} \
--env WANDB_API_KEY=$WANDB_API_KEY --env HostName=$(hostname) --env ServerNum={''.join(devices)} \
{image} {script}
"""

print(text)
os.system(text)