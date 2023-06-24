import os
import sys

import pytorch_lightning as pl
import torch
from clearml import Task
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    LearningRateMonitor,
    ModelCheckpoint
)
from rich import print

# from config.default import TrainingConfig as OldTrainingConfig
from src.data import ImageDataModule
from src.net import Classifier
from src.test import ModelPredictor
from src.utils import (
    export_upload_model, 
    make_graph_performance,
)

from config.config import args_data, args_train, args_model, args_custom
from src.schema.config import (
    CustomConfig, 
    DataConfig, 
    ModelConfig, 
    TrainConfig
)
from src.utils import read_yaml

cwd = os.getcwd()
os.environ["PYTHONPATH"] = cwd
sys.path.append(cwd)

# ----------------------------------------------------------------------------------
# ClearML Setup
# ----------------------------------------------------------------------------------
Task.add_requirements("/workspace/requirements.txt")
task = Task.init(
    project_name="Debug/Data Module",
    task_name="Data Modul",
    task_type=Task.TaskTypes.training,
    auto_connect_frameworks=False,
    tags=["template-v3.0", "debug", "data"],
)
Task.current_task().set_script(
    repository="https://github.com/muhammadAgfian96/pytorch-lighting-image-classifier.git",
    branch="new/v2",
    working_dir=".",
    entry_point="src/train.py",
)
Task.current_task().set_base_docker(
    docker_image="pytorch/pytorch:latest",
    docker_arguments=["--shm-size=8g", "-e PYTHONPATH=/workspace"],
    docker_setup_bash_script=[
        "apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx"
        " libsm6 libxext6 libxrender-dev"
    ],
)
Task.current_task().set_packages(packages="/workspace/requirements.txt")

print(
"""
# ----------------------------------------------------------------------------------
# Manage Configuration
# ----------------------------------------------------------------------------------
"""
)

task.connect(args_data, "1_Data")
task.connect(args_model, "2_Model")
task.connect(args_train, "3_Training")
task.connect(args_custom, "4_Custom")

path_data_yaml = "/workspace/config/datasetsv2.yaml"
path_data_yaml = task.connect_configuration(path_data_yaml, "datasets.yaml")
d_data_yaml = read_yaml(path_data_yaml) 

d_data_config = DataConfig(**args_data)
d_train = TrainConfig(**args_train)
d_model = ModelConfig(**args_model)
d_custom = CustomConfig(**args_custom)

# task.execute_remotely()

print(
"""
# ----------------------------------------------------------------------------------
# Prepare Data, Model, Callbacks For Training 
# ----------------------------------------------------------------------------------
"""
)
pl.seed_everything(32)

print("# Data ---------------------------------------------------------------------")
auto_batch = False
auto_lr_find = False

if d_train.batch == -1:
    auto_batch = True
    d_train.batch = 4
    print("USING AUTO_BATCH")
if d_train.lr == -1:
    d_train.lr = 0.001
    auto_lr_find = True
    print("USING AUTO_LR_FIND")

data_module = ImageDataModule(
    d_train=d_train, 
    d_dataset=d_data_config, 
    d_model=d_model
)
data_module.prepare_data()
d_data_config.num_classes = len(data_module.classes_name)
d_data_config.classes = data_module.classes_name

task.set_model_label_enumeration(d_data_config.label2index())

data_module.visualize_augmented_images("train-augment", num_images=50)
data_module.visualize_augmented_images("val-augment", num_images=15)
# task.mark_completed()
# exit()
print("# Callbacks -----------------------------------------------------------------")
checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    save_last=True,
    save_top_k=1,
    save_weights_only=True,
    verbose=True,
    filename="{epoch}-{val_acc:.2f}-{val_loss:.2f}",
)
checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-{val_acc:.2f}-{val_loss:.2f}-last"

early_stop_callback = EarlyStopping(
    monitor="val_acc",
    min_delta=0.01,
    patience=5,
    verbose=False,
    mode="max",
    check_on_train_epoch_end=True,
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")

# create callbacks
ls_callback = [
    checkpoint_callback,
    lr_monitor,
]


accelerator = "gpu" if torch.cuda.is_available() else "cpu"
model_classifier = Classifier(
    d_train=d_train,
    d_data=d_data_config,
    d_model=d_model
)


print(
    """
# ----------------------------------------------------------------------------------
# Training and Testing 
# ----------------------------------------------------------------------------------
"""
)
trainer = pl.Trainer(
    max_steps=-1,
    max_epochs=d_train.epoch,
    accelerator=accelerator,
    devices=1,
    logger=True,
    callbacks=ls_callback,
    precision=d_train.precision,
    auto_scale_batch_size=auto_batch,
    auto_lr_find=auto_lr_find,
    log_every_n_steps=4,
    # profiler="simple"
)
data_module.setup(stage="fit")
trainer.tune(model=model_classifier, datamodule=data_module)
print(f">> TUNE BATCH_SIZE USE: {data_module.batch_size}")
trainer.fit(model=model_classifier, datamodule=data_module)
print(f">> TUNE BATCH_SIZE USE: {data_module.batch_size}")
# X persen dari autotune batch_size

data_module.setup(stage="test")
trainer.test(datamodule=data_module)
print(f">> TEST TUNE BATCH_SIZE USE: {data_module.batch_size}")


ls_upload_model = [
    {
        "name_upload": "best-ckpt",
        "framework": "Pytorch Lightning",
        "path_weights": checkpoint_callback.best_model_path,
    },
    {
        "name_upload": "lastest-ckpt",
        "framework": "Pytorch Lightning",
        "path_weights": checkpoint_callback.last_model_path,
    },
]
for d_item in ls_upload_model:
    export_upload_model(conf=conf, **d_item)

# Export Model
print(
    """
# ----------------------------------------------------------------------------------
# Export Model 
# ----------------------------------------------------------------------------------
"""
)
input_sample = torch.randn((1, 3, conf.data.input_size, conf.data.input_size))
path_export_model = "export_model"
os.makedirs(path_export_model, exist_ok=True)

path_onnx = os.path.join(path_export_model, f"onnx-{conf.net.architecture}.onnx")
path_torchscript = os.path.join(
    path_export_model, f"torchscript-{conf.net.architecture}.pt"
)

print("Exporting model to TorchScript...")
torch.jit.save(model_classifier.to_torchscript(), path_torchscript)

print("Exporting model to ONNX...")
model_classifier.to_onnx(path_onnx, input_sample)

print(
"""
# ----------------------------------------------------------------------------------
# Testing Model ONNX and TorchScript then upload to 51
# ----------------------------------------------------------------------------------
"""
)


model_tester = ModelPredictor(
    input_size=conf.data.input_size, mean=conf.data.mean, std=conf.data.std
)

model_tester.load_onnx_model(path_onnx)
model_tester.load_torchscript_model(path_torchscript)

if data_module.ls_test_map_dedicated is not None:
    d_onnx_51 = model_tester.predict_onnx_dataloaders(
        dataloaders=data_module.ls_test_map_dedicated,
        classes=conf.data.category,
    )
    d_torchscript_51 = model_tester.predict_torchscript_dataloaders(
        dataloaders=data_module.ls_test_map_dedicated,
        classes=conf.data.category,
    )

    Task.current_task().upload_artifact(
        "onnx_test_51",
        d_onnx_51,
    )
    Task.current_task().upload_artifact("torchscript_test_51", d_torchscript_51)

    print("torchscript:", d_torchscript_51["info"])
    print("onnx:", d_onnx_51["info"])

    fig_performance = make_graph_performance(
        torchscript_performance=d_torchscript_51["info"],
        onnx_performance=d_onnx_51["info"],
    )
    Task.current_task().get_logger().report_plotly(
        series="Performance ONNX vs TorchScript",
        title="Performance",
        iteration=0,
        figure=fig_performance,
    )


print(
    """
# ----------------------------------------------------------------------------------
# Reporting 
# ----------------------------------------------------------------------------------
"""
)
ls_upload_model = [
    {"name_upload": "onnx", "framework": "ONNX", "path_weights": path_onnx},
    {
        "name_upload": "torchscript",
        "framework": "Pytorch",
        "path_weights": path_torchscript,
    },
]

for d_item in ls_upload_model:
    export_upload_model(conf=conf, **d_item)
