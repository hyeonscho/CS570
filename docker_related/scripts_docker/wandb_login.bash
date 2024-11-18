# !/bin/bash
# doesnot work inside docker
# ####!/bin/sh
# export WANDB_API_KEY=$(cat /scripts/.wandb_api_key)
# export WANDB_API_KEY=$(cat $(pwd)/scripts/.wandb_api_key)
# working
WANDB_API_KEY=$(cat $(pwd)/scripts_docker/.wandb_api_key)
# Not working
# echo $WANDB_API_KEY
# export WANDB_API_KEY=$(cat /scripts/.wandb_api_key)
# echo $WANDB_API_KEY
# -------------
# export test_hi=$(cat /scripts/.wandb_api_key)