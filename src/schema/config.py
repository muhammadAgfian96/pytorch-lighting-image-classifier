import os
from pydantic import BaseModel
from typing import Optional, List

curr_path = os.getcwd()


class TrainConfig(BaseModel):
    epoch:int = 10
    batch:int = -1
    optimizer:str = "AdamW"
    weight_decay:float
    momentum:float
    lr:float
    lr_scheduler:str
    lr_step_size:int
    lr_decay_rate:int
    precision:int = 16

class ModelConfig(BaseModel):
    input_size:int
    architecture:str
    dropout:float
    path_pretrained:Optional[str] = None

class CustomConfig(BaseModel):
    tags_exclude:List[str]


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
    yaml_path:str = os.path.join(curr_path, "config/datasetsv2.yaml")

    mean:Optional[List[float]] = None 
    std:Optional[List[float]] = None

    classes: Optional[List[str]]
    num_classes: Optional[int]

    def label2index(self):
        return {lbl: idx for idx, lbl in enumerate(self.classes)}


