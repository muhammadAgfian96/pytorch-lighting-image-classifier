import os

import lightning.pytorch as pl
import torch
from clearml import Task
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.tuner import Tuner

import src.env  # load environment variables
from src.data import ImageDataModule
from src.net import Classifier
from src.net_v2 import ModelClassifier
from src.utils.callbacks import CallbackClearML
from src.utils.clearml import clearml_configuration, clearml_init
from src.utils.logger import LoggerStdOut


log = LoggerStdOut()

# ClearML Setup
log.title_section('ClearML Setup')
task = clearml_init()

log.title_section('Configuration')
d_data_config, d_train, d_model, d_custom = clearml_configuration()

# task.execute_remotely()

log.title_section("Prepare Data, Model, Callbacks For Training")
pl.seed_everything(os.getenv("RANDOM_SEED", 32))

log.sub_section("Setup Data")
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
)

log.sub_section("Tuning Params")
# tuner = Tuner(trainer)
# tuner.scale_batch_size(model_classifier, datamodule=data_module, mode="binsearch")
# tuner.lr_find(model_classifier, datamodule=data_module, mode="linear")

data_module.setup(stage="fit")
# trainer.tune(model=model_classifier, datamodule=data_module)
print(f">> TUNE BATCH_SIZE USE: {data_module.batch_size}")
trainer.fit(model=model_classifier, datamodule=data_module)
print(f">> TUNE BATCH_SIZE USE: {data_module.batch_size}")
# X persen dari autotune batch_size

data_module.setup(stage="test")
trainer.test(datamodule=data_module)
print(f">> TEST TUNE BATCH_SIZE USE: {data_module.batch_size}")


# ls_upload_model = [
#     {
#         "name_upload": "best-ckpt",
#         "framework": "Pytorch Lightning",
#         "path_weights": checkpoint_callback.best_model_path,
#     },
#     {
#         "name_upload": "lastest-ckpt",
#         "framework": "Pytorch Lightning",
#         "path_weights": checkpoint_callback.last_model_path,
#     },
# ]

# for d_item in ls_upload_model:
#     export_upload_model(conf=conf, **d_item)

# # Export Model
# print(
#     """
# # ----------------------------------------------------------------------------------
# # Export Model 
# # ----------------------------------------------------------------------------------
# """
# )
# input_sample = torch.randn((1, 3, conf.data.input_size, conf.data.input_size))
# path_export_model = "export_model"
# os.makedirs(path_export_model, exist_ok=True)

# path_onnx = os.path.join(path_export_model, f"onnx-{conf.net.architecture}.onnx")
# path_torchscript = os.path.join(
#     path_export_model, f"torchscript-{conf.net.architecture}.pt"
# )

# print("Exporting model to TorchScript...")
# torch.jit.save(model_classifier.to_torchscript(), path_torchscript)

# print("Exporting model to ONNX...")
# model_classifier.to_onnx(path_onnx, input_sample)

# print(
# """
# # ----------------------------------------------------------------------------------
# # Testing Model ONNX and TorchScript then upload to 51
# # ----------------------------------------------------------------------------------
# """
# )


# model_tester = ModelPredictor(
#     input_size=conf.data.input_size, mean=conf.data.mean, std=conf.data.std
# )

# model_tester.load_onnx_model(path_onnx)
# model_tester.load_torchscript_model(path_torchscript)

# if data_module.ls_test_map_dedicated is not None:
#     d_onnx_51 = model_tester.predict_onnx_dataloaders(
#         dataloaders=data_module.ls_test_map_dedicated,
#         classes=conf.data.category,
#     )
#     d_torchscript_51 = model_tester.predict_torchscript_dataloaders(
#         dataloaders=data_module.ls_test_map_dedicated,
#         classes=conf.data.category,
#     )

#     Task.current_task().upload_artifact(
#         "onnx_test_51",
#         d_onnx_51,
#     )
#     Task.current_task().upload_artifact("torchscript_test_51", d_torchscript_51)

#     print("torchscript:", d_torchscript_51["info"])
#     print("onnx:", d_onnx_51["info"])

#     fig_performance = make_graph_performance(
#         torchscript_performance=d_torchscript_51["info"],
#         onnx_performance=d_onnx_51["info"],
#     )
#     Task.current_task().get_logger().report_plotly(
#         series="Performance ONNX vs TorchScript",
#         title="Performance",
#         iteration=0,
#         figure=fig_performance,
#     )


# log.title_section("Reporting")
# ls_upload_model = [
#     {"name_upload": "onnx", "framework": "ONNX", "path_weights": path_onnx},
#     {
#         "name_upload": "torchscript",
#         "framework": "Pytorch",
#         "path_weights": path_torchscript,
#     },
# ]

# for d_item in ls_upload_model:
#     export_upload_model(conf=conf, **d_item)
