import os
from collections import defaultdict
from dataclasses import asdict
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import timm
import torch
import torch.optim as optim
from clearml import InputModel, Task
from PIL import Image
from rich import print
from sklearn.metrics import precision_recall_fscore_support
import torchmetrics as tm
from config.list_models import list_models as ls_models_library
from src.schema.config import DataConfig, ModelConfig, TrainConfig
from utils.utils_roc import generate_plot_one_vs_one, generate_plot_one_vs_rest
from schema.net_v2 import OutputStep

class ModelCreation:
    def __init__(self, architecture:str, num_classes:int, dropout:float) -> None:
        self.architecture = architecture
        self.path_pretrained = None
        self.num_classes = num_classes
        self.num_classes = dropout

        print(timm.list_models())
        print(architecture not in timm.list_models())
        if architecture not in timm.list_models():
            model_old = InputModel(model_id=self.architecture)
            self.architecture = model_old.config_dict["net"]
            print("dowloding model", model_old.config_dict)
            print("old model labels", model_old.labels)
            model_old_path = model_old.get_weights()
            self.path_pretrained = model_old_path

        self.model = timm.create_model(
            self.architecture,
            pretrained=self.path_pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

        if self.path_pretrained is not None:
            # raw way
            checkpoint = torch.load(self.d_model.path_pretrained)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            Task.current_task().add_tags("resume")
    
    # def __call__(self, x):
    #     return self.model(x)


class ModelClassifier(pl.LightningModule):
    def __init__(self, d_train:TrainConfig, d_data:DataConfig, d_model:ModelConfig) -> None:
        super().__init__()
        self.model = ModelCreation(
            architecture=d_model.architecture,
            num_classes=d_data.num_classes,
            dropout=d_model.dropout
        ).model
        self.d_train = d_train
        self.d_data = d_data
        self.d_model = d_model


        self.learning_rate = d_train.lr

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=d_data.num_classes)
        self.train_f1 = tm.F1Score(task="multiclass", num_classes=d_data.num_classes)
        self.val_acc = tm.Accuracy(task="multiclass", num_classes=d_data.num_classes)
        self.val_f1 = tm.F1Score(task="multiclass", num_classes=d_data.num_classes)
        self.test_acc = tm.Accuracy(task="multiclass", num_classes=d_data.num_classes)
        self.test_f1 = tm.F1Score(task="multiclass", num_classes=d_data.num_classes)

        self.output_train_step = OutputStep()
        self.output_val_step = OutputStep()
        self.output_test_step = OutputStep()



    def configure_optimizers(self):
        opt_d = {
            "Adam": optim.Adam(self.model.parameters(), lr=self.learning_rate),
            "SGD": optim.SGD(self.model.parameters(), lr=self.learning_rate),
            "AdamW": optim.AdamW(self.model.parameters(), lr=self.learning_rate),
            "RMSprop": optim.RMSprop(self.model.parameters(), lr=self.learning_rate),
            "Nadam": optim.NAdam(self.model.parameters(), lr=self.learning_rate),
            "Adadelta": optim.Adadelta(self.model.parameters(), lr=self.learning_rate),
            "Adagrad": optim.Adagrad(self.model.parameters(), lr=self.learning_rate),
            "Adamax": optim.Adamax(self.model.parameters(), lr=self.learning_rate),
        }

        optimizer = opt_d.get(
            self.d_train.optimizer,
            optim.Adam(self.model.parameters(), lr=self.learning_rate),
        )
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.output_train_step.add(loss, preds, labels)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.output_val_step.add(loss, preds, labels)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.output_test_step.add(loss, preds, labels)
        return loss

    def _common_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.forward(images)
        loss = self.loss_fn(preds, labels)
        return loss, preds, labels
    
    def predict_step(self, batch, batch_idx):
        images, labels = batch
        scores = self.forward(images)
        preds = torch.argmax(scores, dim=1)
        return preds



    
