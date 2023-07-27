import os
from typing import Tuple

from clearml import Task

from config.config import args_custom, args_data, args_model, args_train
from src.schema.config import (
    CustomConfig, 
    DataConfig, 
    ModelConfig,
    TrainConfig
)
from src.utils.utils import read_yaml
from rich import print

def clearml_init() -> Task:
    req_path = os.path.join(os.getcwd(), 'requirements.txt')
    tags = ["🏷️ v2.1"]
    if os.getenv("MODE_TEMPLATE", "remote") == "debug":
        tags.append("debug")


    Task.add_requirements(req_path)
    task:Task = Task.init(
        project_name="Debug/New Template",
        task_name="Image Classifier",
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks=False,
        reuse_last_task_id=False,
    )
    task.set_tags(tags)
    task.set_script(
        repository="https://github.com/muhammadAgfian96/pytorch-lighting-image-classifier.git",
        branch="new/v2",
        working_dir=".",
        entry_point="src/train.py",
    )
    task.set_base_docker(
        docker_image="torch-classifier:latest",
        docker_arguments=["--shm-size=8g", "-e PYTHONPATH=/workspace"]
    )
    return Task.current_task()

def clearml_configuration() -> Tuple[DataConfig, TrainConfig, ModelConfig, CustomConfig]:
    task:Task = Task.current_task()

    task.connect(args_data, "1_Data")
    task.connect(args_model, "2_Model")
    task.connect(args_train, "3_Training")
    task.connect(args_custom, "4_Custom")

    # update dataset yaml via clearml ui
    path_data_yaml = os.path.join(os.getcwd(), "config/datasets.yaml")
    print(f"data_sebelum {path_data_yaml}")
    path_data_yaml = task.connect_configuration(path_data_yaml, "datasets.yaml")
    print(f"data_sesudah: {path_data_yaml}")

    d_data_config = DataConfig(**args_data, yaml_path=path_data_yaml)
    d_train = TrainConfig(**args_train)
    d_model = ModelConfig(**args_model)
    d_custom = CustomConfig(**args_custom)

    return d_data_config, d_train, d_model, d_custom
