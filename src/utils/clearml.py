import os
from clearml import Task

from config.config import args_custom, args_data, args_model, args_train
from src.schema.config import (CustomConfig, DataConfig, ModelConfig,
                               TrainConfig)
from src.utils.utils import read_yaml
from typing import Tuple



def clearml_init() -> Task:
    req_path = os.path.join(os.getcwd(), 'requirements.txt')
    tags = ["template-v4.0"]
    if os.getenv("MODE_TEMPLATE", "remote") == "debug":
        tags.append("debug")


    Task.add_requirements(req_path)
    task:Task = Task.init(
        project_name="Debug/Data Module",
        task_name="Data Modul",
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks=False,
        tags=tags,
    )
    task.set_script(
        repository="https://github.com/muhammadAgfian96/pytorch-lighting-image-classifier.git",
        branch="new/v2",
        working_dir=".",
        entry_point="src/train.py",
    )
    task.set_base_docker(
        docker_image="pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime",
        docker_arguments=["--shm-size=8g", "-e PYTHONPATH=/workspace"],
        docker_setup_bash_script=[
            "apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx"
            " libsm6 libxext6 libxrender-dev"
        ],
    )
    return Task.current_task()

def clearml_configuration() -> Tuple[DataConfig, TrainConfig, ModelConfig, CustomConfig]:
    task:Task = Task.current_task()

    task.connect(args_data, "1_Data")
    task.connect(args_model, "2_Model")
    task.connect(args_train, "3_Training")
    task.connect(args_custom, "4_Custom")

    path_data_yaml = os.path.join(os.getcwd(),"config/datasetsv2.yaml")
    path_data_yaml = task.connect_configuration(path_data_yaml, "datasets.yaml")
    d_data_yaml = read_yaml(path_data_yaml) 

    d_data_config = DataConfig(**args_data)
    d_train = TrainConfig(**args_train)
    d_model = ModelConfig(**args_model)
    d_custom = CustomConfig(**args_custom)

    return d_data_config, d_train, d_model, d_custom