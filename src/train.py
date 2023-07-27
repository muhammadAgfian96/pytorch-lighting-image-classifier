import os

import lightning.pytorch as pl
import torch
from clearml import OutputModel, Task
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.tuner import Tuner

import src.env  # load environment variables
from src.data import ImageDataModule
from src.net import Classifier
from src.net_v2 import ModelClassifier
from src.utils.callbacks import CallbackClearML
from src.utils.clearml import clearml_configuration, clearml_init
from src.utils.export import export_handler
from src.utils.logger import LoggerStdOut
import math
log = LoggerStdOut()

# ClearML Setup
log.title_section('ClearML Setup')
task = clearml_init()

log.title_section('Configuration Setup')
d_data_config, d_train, d_model, d_custom = clearml_configuration()

# task.execute_remotely()

log.title_section("Prepare Data, Model, Callbacks For Training")
pl.seed_everything(os.getenv("RANDOM_SEED", 32))

log.sub_section("Setup Data")
data_module = ImageDataModule(
    d_train=d_train, 
    d_dataset=d_data_config, 
    d_model=d_model,
    d_custom=d_custom
)
data_module.prepare_data()
d_data_config.num_classes = len(data_module.classes_name)
d_data_config.classes = data_module.classes_name

if d_custom.mode == "testing":
    task.set_task_type(Task.TaskTypes.testing)
    
task.set_model_label_enumeration(d_data_config.label2index())

data_module.visualize_augmented_images("train-augment", num_images=50)
data_module.visualize_augmented_images("val-augment", num_images=15)

log.sub_section("Setup Callbacks")
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


lr_monitor = LearningRateMonitor(logging_interval="step")

# create callbacks
ls_callback = [
    checkpoint_callback,
    lr_monitor,
    CallbackClearML()
]

if d_train.early_stopping.patience > 0:
    early_stop_callback = EarlyStopping(
        monitor=d_train.early_stopping.monitor,
        min_delta=d_train.early_stopping.min_delta,
        mode=d_train.early_stopping.mode,
        patience=d_train.early_stopping.patience,
        verbose=True,
        check_on_train_epoch_end=True,
    )
    ls_callback.append(early_stop_callback)

log.sub_section("Prepare Model")
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
model_classifier = ModelClassifier(
    d_train=d_train,
    d_data=d_data_config,
    d_model=d_model
)


log.title_section("Training and Testing")
trainer = pl.Trainer(
    accelerator="auto",
    devices="auto",
    max_steps=-1,
    max_epochs=d_train.epoch,
    logger=True,
    callbacks=ls_callback,
    precision=d_train.precision,
    default_root_dir="./tmp",
    log_every_n_steps=10
)

if d_custom.mode == "training":
    log.sub_section("Tuning Params")
    tuner = Tuner(trainer)
    if d_train.tuner.batch_size:
        log.sub_section("BATCH SIZE TUNING (AUTO SEARCH)")
        try:
            batch_size_finder = tuner.scale_batch_size(
                model_classifier, 
                # method="fit",
                datamodule=data_module, 
                mode="binsearch",
                steps_per_trial=6,
                max_trials=5,
                init_val=d_train.batch if d_train.batch <= 32 else 4
            )
            if batch_size_finder > 100:
                data_module.batch_size = math.ceil(batch_size_finder*0.8)
                print(f"💡 batch_size_finder: {batch_size_finder} -> batch_size: {data_module.batch_size}")
            Task.current_task().set_parameter("_autoset/batch_size_finder", data_module.batch_size)
            Task.current_task().set_parameter("3_Training/batch", data_module.batch_size)
        except Exception as e:
            print("Error autobatch:")
            print(e)

    if d_train.tuner.learning_rate:
        log.sub_section("Learning Rate TUNING (AUTO SEARCH)")

        lr_finder = tuner.lr_find(
            model_classifier, 
            datamodule=data_module, 
            mode="exponential",
            min_lr=1e-5,
            max_lr=1e-1,
            early_stop_threshold=5,
            attr_name="learning_rate"
        )
        fig = lr_finder.plot(suggest=True)
        Task.current_task().get_logger().report_matplotlib_figure("lr_finder", "lr_finder", iteration=0, figure=fig)
        Task.current_task().set_parameter("_autoset/lr_finder", lr_finder.suggestion())

    log.sub_section("Training Start")
    data_module.setup(stage="fit")
    trainer.fit(model=model_classifier, datamodule=data_module)

    data_module.setup(stage="test")
    trainer.test(datamodule=data_module)

elif d_custom.mode == "testing":
    log.sub_section("Testing Only")

    data_module.setup(stage="test")
    trainer.test(datamodule=data_module, model=model_classifier)

if d_custom.mode == "training" and os.getenv("DEBUG", "no") == "no":
    export_handler(
        checkpoint_callback=checkpoint_callback, 
        model_classifier=model_classifier, 
        d_data=d_data_config, 
        d_model=d_model, 
    )

