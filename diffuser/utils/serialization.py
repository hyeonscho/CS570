import os
import pickle
import glob
import torch
import pdb

from collections import namedtuple

DiffusionExperiment = namedtuple(
    "Diffusion", "dataset renderer model diffusion ema trainer epoch"
)



ClassifierExperiment = namedtuple(
    "Classifier", "dataset renderer model ema trainer epoch"
)

def mkdir(savepath):
    """
    returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False


def get_all_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), "state_*")
    epochs = []
    for state in states:
        epoch = int(state.replace("state_", "").replace(".pt", ""))
        epochs.append(epoch)
    return epochs


def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), "state_*")
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace("state_", "").replace(".pt", ""))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch


def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, "rb"))
    print(f"[ utils/serialization ] Loaded config from {loadpath}")
    print(config)
    return config


def load_diffusion(*loadpath, epoch="latest", device="cuda:0"):
    # loadpath has some parameters which is a list, and I want to convert it to a string similar to the saved path way
    loadpath = list(loadpath)
    for i in range(len(loadpath)):
        loadpath[i] = loadpath[i].replace(', ', '-')
    dataset_config = load_config(*loadpath, "dataset_config.pkl")
    render_config = load_config(*loadpath, "render_config.pkl")
    model_config = load_config(*loadpath, "model_config.pkl")
    diffusion_config = load_config(*loadpath, "diffusion_config.pkl")
    trainer_config = load_config(*loadpath, "trainer_config.pkl")

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict["results_folder"] = os.path.join(*loadpath)

    dataset = dataset_config()
    renderer = render_config()
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)

    if epoch == "latest":
        epoch = get_latest_epoch(loadpath)

    if epoch != -1:
        print(f"\n[ utils/serialization ] Loading model epoch: {epoch}\n")

        trainer.load(epoch)

    return DiffusionExperiment(
        dataset, renderer, model, diffusion, trainer.ema_model, trainer, trainer.step
    )


def load_classifier(*loadpath, epoch="latest", device="cuda:0"):
    # loadpath has some parameters which is a list, and I want to convert it to a string similar to the saved path way
    loadpath = list(loadpath)
    for i in range(len(loadpath)):
        loadpath[i] = loadpath[i].replace(', ', '-')

    dataset_config = load_config(*loadpath, "dataset_config.pkl")
    render_config = load_config(*loadpath, "render_config.pkl")
    classifier_config = load_config(*loadpath, "classifier_config.pkl")
    trainer_config = load_config(*loadpath, "trainer_config.pkl")

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict["results_folder"] = os.path.join(*loadpath)

    dataset = dataset_config()
    renderer = render_config()
    model = classifier_config()
    trainer = trainer_config(model, dataset, renderer)

    if epoch == "latest":
        epoch = get_latest_epoch(loadpath)

    if epoch != -1:
        print(f"\n[ utils/serialization ] Loading model epoch: {epoch}\n")

        trainer.load(epoch)

    return ClassifierExperiment(
        dataset, renderer, model, trainer.ema_model, trainer, trainer.step
    )
