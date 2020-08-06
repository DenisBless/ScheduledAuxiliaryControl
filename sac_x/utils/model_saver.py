import os
import datetime
import torch
import pathlib


class ModelSaver:
    def __init__(self, model_root_dir=None, save_every=10):
        if model_root_dir is None:
            model_root_dir = str(pathlib.Path(__file__).resolve().parents[2]) + "/models/"

        if not os.path.isdir(model_root_dir):
            os.mkdir(model_root_dir)
        self.model_dir = model_root_dir + datetime.datetime.now().strftime("%d-%m_%H-%M")

        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

        self.save_every = save_every
        self.ctr = 0

    def save_model(self, model, name: str):
        if self.ctr != 0 and self.ctr % self.save_every == 0:
            torch.save(model, self.model_dir + "/" + name + "_" + str(self.ctr))
        self.ctr += 1
