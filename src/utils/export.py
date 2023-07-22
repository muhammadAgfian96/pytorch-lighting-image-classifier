import torch
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from src.utils.logger import LoggerStdOut
from clearml import OutputModel, Task
import os
from src.schema.config import (CustomConfig, DataConfig, ModelConfig,
                               TrainConfig)
from src.net_v2 import ModelClassifier
from src.test import ModelPredictor
log = LoggerStdOut()


def export_upload_model(d_model: ModelConfig,
    name_upload: str,
    framework: str,
    path_weights: str,
):
    """
    Export model to ONNX and TorchScript then upload to 51
    """

    try:
        print(f"Uploading {name_upload}-{framework} >>>".upper())
        output_model = OutputModel(
            task=Task.current_task(),
            name=name_upload,
            framework=framework,
        )
        extenstion = os.path.basename(path_weights).split(".")[-1]
        output_model.update_weights(
            weights_filename=path_weights,
            target_filename=f"{name_upload}-{d_model.architecture}.{extenstion}",  # it will name output
        )
        output_model.update_design(
            config_dict={
                "net": d_model.architecture,
                "input_size": d_model.input_size,
            }
        )
        output_model.set_metadata("imgsz", value=d_model.input_size)
        output_model.set_metadata("architecture", value=d_model.architecture)
    except Exception as e:
        print(f"Error Upload {name_upload}-{framework}".upper(), e)


def export_handler(
        checkpoint_callback: ModelCheckpoint, 
        d_model: ModelConfig,
        model_classifier: ModelClassifier,
        d_data: DataConfig,
    ):

    path_export_model = "export_model"
    os.makedirs(path_export_model, exist_ok=True)

    input_sample = torch.randn((1, 3, d_model.input_size, d_model.input_size))


    ls_upload_model = [
        {
            "name_upload": "best-raw",
            "framework": "Pytorch",
            "path_weights": checkpoint_callback.best_model_path,
        },
        {
            "name_upload": "lastest-raw",
            "framework": "Pytorch",
            "path_weights": checkpoint_callback.last_model_path,
        },
        {
            "name_upload": "torchscript",
            "framework": "Pytorch",
            "path_weights": os.path.join(path_export_model, f"torchscript-{d_model.architecture}.pt"),
        },
        {
            "name_upload": "onnx",
            "framework": "ONNX",
            "path_weights": os.path.join(path_export_model, f"onnx-{d_model.architecture}.onnx"),
        },
    ]

    # Export Model
    log.title_section("Export Model")

    log.sub_section("Upload raw model")
    for d_item in ls_upload_model:
        try:
            if d_item["name_upload"] == "onnx":
                print("Exporting model to ONNX...")
                model_classifier.to_onnx(d_item["path_weights"], input_sample)

            if d_item["name_upload"] == "torchscript":
                print("Exporting model to TorchScript...")
                torch.jit.save(model_classifier.to_torchscript(), d_item["path_weights"])

            export_upload_model(d_model=d_model, **d_item)
        except Exception as e:
            print("ERROR to EXPORT MODEL")
            print(d_item)
            print(e)


def testing_model_prediction(
        d_model: ModelConfig, 
        d_data: DataConfig, 
        data_module, 
        path_onnx: str, 
        path_torchscript: str
    ):
    log.title_section("Testing Model")

    model_tester = ModelPredictor(
        input_size=d_model.input_size, mean=d_data.mean, std=d_data.std
    )

    model_tester.load_onnx_model(path_onnx)
    model_tester.load_torchscript_model(path_torchscript)

    if data_module.ls_test_map_dedicated is not None:
        d_onnx_51 = model_tester.predict_onnx_dataloaders(
            dataloaders=data_module.ls_test_map_dedicated,
            classes=d_data.classes,
        )
        d_torchscript_51 = model_tester.predict_torchscript_dataloaders(
            dataloaders=data_module.ls_test_map_dedicated,
            classes=d_data.classes,
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


    log.title_section("Reporting")
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
