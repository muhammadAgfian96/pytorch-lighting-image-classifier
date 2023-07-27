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
from timm.optim.optim_factory import create_optimizer, create_optimizer_v2
from timm.scheduler.scheduler_factory import create_scheduler_v2
from config.list_timm_of_optimizer import LIST_OPTIMIZERS_TIMM
import timm
import uuid

def is_uuid(name):
    # Check if the name is a timm model
    # name_model = name.replace("timm/", "")
    # if name_model in timm.list_models():
    #     return True
    
    # Check if the name is a valid UUID
    try:
        uuid.UUID(name, version=4)
        return True
    except ValueError:
        return False

def is_timm_model(name):

    name_model = name.replace("timm/", "")

    try:
        timm.create_model(name_model)
        return True
    except Exception as e:
        print(f"ERROR Load Model via Timm {name}")
        print(e)
        return False


class ModelCreation:
    def __init__(self, architecture:str, num_classes:int, dropout:float, resume:str=None) -> None:
        self.architecture = architecture
        self.path_pretrained = None
        self.num_classes = num_classes
        self.num_classes = dropout
        self.clearml_id_resume = resume

        # print(timm.list_models())
        _is_uuid = is_uuid(self.clearml_id_resume)
        _is_timm_model = is_timm_model(self.architecture)
        print("architecture in timm_model:", _is_timm_model, "| is_uuid:", _is_uuid)

        if _is_uuid is False and _is_timm_model is False:
            Task.current_task().add_tags("⛔:architecture")
            raise Exception("check again your model architecture or model_clearml_id")
        
        if _is_uuid and not _is_timm_model:
            model_old = InputModel(model_id=self.clearml_id_resume)
            self.architecture = model_old.config_dict["net"]
            print("model: ", self.architecture)
            print("dowloding model", model_old.config_dict)
            print("old model labels", model_old.labels)
            model_old_path = model_old.get_weights()
            self.path_pretrained = model_old_path
            Task.current_task().set_parameter(name="2_Model/architecture", value=self.architecture)

        self.model = timm.create_model(
            self.architecture,
            pretrained=self.path_pretrained or True,
            num_classes=num_classes,
            drop_rate=dropout,
        )

        if self.path_pretrained is not None:
            # raw way
            checkpoint = torch.load(self.path_pretrained)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            Task.current_task().add_tags("▶️")
    
    # def __call__(self, x):
    #     return self.model(x)


class ModelClassifier(pl.LightningModule):
    def __init__(self, d_train:TrainConfig, d_data:DataConfig, d_model:ModelConfig) -> None:
        super().__init__()
        self.initial_vram_memory = torch.cuda.max_memory_allocated() / 1024 ** 2  # Convert to MB

        self.model_creation = ModelCreation(
            architecture=d_model.architecture,
            num_classes=d_data.num_classes,
            dropout=d_model.dropout,
            resume=d_model.resume
        )
        self.model = self.model_creation.model
        self.model_architecture = self.model_creation.architecture

        self.final_vram_memory = torch.cuda.max_memory_allocated() / 1024 ** 2  # Convert to MB

        self.d_train = d_train
        self.d_data = d_data
        self.d_model = d_model
        self.d_model.architecture = self.model_architecture

        self.learning_rate = d_train.lr

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.output_train_step = OutputStep()
        self.output_val_step = OutputStep()
        self.output_test_step = OutputStep()



    def configure_optimizers(self):
        opt_name = self.d_train.optimizer.lower()
        if opt_name not in LIST_OPTIMIZERS_TIMM:
            opt_name = "adamw"
        
        optimizer = create_optimizer_v2(
            model_or_params=self.model.parameters(), 
            opt=opt_name, 
            lr=self.learning_rate
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
        imgs, y = batch
        self.output_test_step.add(loss, preds, labels, imgs)
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



    
