from pydantic import BaseModel
from typing import Optional, List

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

class DataConfig(BaseModel):
    augment:str = "custom"
    ratio_train:float
    ratio_val:float
    ratio_test:float

    dir_output:str = "/workspace/dataset"
    yaml_path:str = "/workspace/config/datasetsv2.yaml"

    mean:Optional[List[float]] = None 
    std:Optional[List[float]] = None

    classes: Optional[List[str]]
    num_classes: Optional[int]

    def label2index(self):
        return {lbl: idx for idx, lbl in enumerate(self.classes)}


