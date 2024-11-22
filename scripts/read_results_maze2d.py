import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import glob
import numpy as np
import json
import pdb
import diffuser.utils as utils

DATASETS = [
    f"{env}-{buffer}-v2"
    for env in ["maze2d"]
    for buffer in ["umaze", "medium", "large"]
]

LOGBASE = "logs"
TRIAL = "*"
EXP_NAME = "plans*/*"
verbose = False


def load_results(paths):
    """
    paths : path to directory containing experiment trials
    """
    scores = []
    returns = []
    for i, path in enumerate(sorted(paths)):
        score, r = load_result(path)
        if verbose:
            print(path, score)
        if score is None:
            # print(f'Skipping {path}')
            continue
        scores.append(score)
        returns.append(r)

        suffix = path.split("/")[-1]
        # print(suffix, path, score)

    num_ = len(scores)
    if len(scores) > 0:
        if len(scores) > 100:
            scores = np.stack(
                [np.array(scores)[idx : idx + 100] for idx in range(num_ - 100)]
            )
            mean = scores.mean(-1).max()
            idx = scores.mean(-1).argmax()
            scores = scores[idx]
            returns = np.stack(
                [np.array(returns)[idx : idx + 100] for idx in range(num_ - 100)]
            )
            returns = returns[idx]
        else:
            mean = np.mean(scores)
            returns = np.array(returns)
        sus_rate = len(np.where(scores)[0]) / len(scores)
    else:
        mean = np.nan
        sus_rate = np.nan

    if len(scores) > 1:
        err = np.std(scores) / np.sqrt(len(scores))
    else:
        err = 0
    return mean, err, scores, sus_rate


def load_result(path):
    """
    path : path to experiment directory; expects `rollout.json` to be in directory
    """
    # fullpath = os.path.join(path, 'rollout.json')

    if not os.path.exists(path):
        return None

    results = json.load(open(path, "rb"))
    score = results["score"] * 100
    r = np.sum(results["return"])
    return score, r


#######################
######## setup ########
#######################

if __name__ == "__main__":
    # configs = ["config.maze2d_hl_hmdConfig300_1M", "config.maze2d_hl_hmdConfig500_1M",
    #            "config.maze2d_hmdConfig300_1M", "config.maze2d_hmdConfig500_1M"]
    # configs = ["config.maze2d_hl_hmdConfig300", "config.maze2d_hl_hmdConfig500",
    #            "config.maze2d_hmdConfig300", "config.maze2d_hmdConfig500"]

    # configs = ["config.maze2d_hl_300",
    #            "config.maze2d_hl_hmdConfig300_J15"]
    
    configs = ['config.maze2d_300', 'config.maze2d_390', 'config.maze2d_hmdConfig300', 'config.maze2d_hmdConfig500', 'config.maze2d_hl_300', 'config.maze2d_hl_390J30', 'config.maze2d_hl_hmdConfig300_J15_OriginalConfigKernel', 'config.maze2d_hl_hmdConfig300_J15', 'config.maze2d_hl_hmdConfig300', 'config.maze2d_hl_hmdConfig390', 'config.maze2d_hl_hmdConfig500_J20', 'config.maze2d_hl_hmdConfig500']

    for cfg in configs:

        class Parser(utils.Parser):
            dataset: str = "maze2d-large-v1"
            config: str = cfg

        args = Parser().parse_args("plan")
        restricted_pd = True
        args.restricted_pd =  restricted_pd
        args.savepath = args.savepath.replace('rpdFalse', f'rpd{restricted_pd}')
        args.savepath = args.savepath.replace('rpdTrue', f'rpd{restricted_pd}')
        print(args.savepath)

        epochs = ["latest"]

        for dataset in [args.dataset] if args.dataset else DATASETS:
            subdir = "/root/diffuser_chain_hd/" + os.path.join(*args.savepath.split("/")[:-1])
            # subdir = subdir.replace("plans/", "plans_1M/")
            # subdir = '/root/hd/logs/maze2d-large-v1/plans/latest/release_H500_T100_LimitsNormalizer_b1_condFalse_J50'
            reldir = subdir.split("/")[-1]
            # print(subdir)
            # print(os.path.join(subdir, TRIAL, f"*_rollout.json"))
            paths = glob.glob(os.path.join(subdir, TRIAL, f"*_rollout.json"))
            paths = sorted(paths)

            mean, err, scores, sus_rate = load_results(paths)
            if np.isnan(mean):
                continue
            path, name = os.path.split(subdir)
            print(
                f"{dataset.ljust(30)} | {name.ljust(50)} | {path.ljust(50)} | {len(scores)} scores \n    {mean:.1f} +/- {err:.2f}"
                f"\nsus_rate: {sus_rate * 100:.2f}"
            )
            if verbose:
                print(scores)
                print(sus_rate)
