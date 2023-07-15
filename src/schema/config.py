import os
from typing import List, Optional, Union

from pydantic import BaseModel

curr_path = os.getcwd()

class EarlyStopping(BaseModel):
    patience:int 
    min_delta:float
    mode:str
    monitor:str

class TunerConfig(BaseModel):
    batch_size:bool
    learning_rate:bool

class TrainConfig(BaseModel):
    epoch:int = 10
    batch:Union[int, str] = "auto"
    optimizer:str = "AdamW"
    weight_decay:float
    momentum:float
    lr:float
    lr_scheduler:str
    lr_step_size:int
    lr_decay_rate:int
    precision:Union[int, str] = 16
    early_stopping: EarlyStopping
    tuner: TunerConfig

class ModelConfig(BaseModel):
    input_size:int
    architecture:str
    dropout:float
    path_pretrained:Optional[str] = None
    resume:Optional[str]

class CustomConfig(BaseModel):
    tags_exclude:List[str]
    mode:str


class AugmentPreset(BaseModel):
    hflip_prob:float
    ra_magnitude:float
    augmix_severity:float
    random_erase_prob:float

class DataConfig(BaseModel):
    ratio_train:float
    ratio_val:float
    ratio_test:float
    augment_type:str
    augment_preset_train: Optional[AugmentPreset]

    dir_dataset_train:str = os.path.join(curr_path, "dataset-train")
    dir_dataset_test:str = os.path.join(curr_path, "dataset-test")
    yaml_path:str = os.path.join(curr_path, "config/datasets.yaml")

    mean:Optional[List[float]] = [0.485, 0.456, 0.406]
    std:Optional[List[float]] = [0.229, 0.224, 0.225]

    classes: Optional[List[str]]
    num_classes: Optional[int]

    def label2index(self):
        return {lbl: idx for idx, lbl in enumerate(self.classes)}


