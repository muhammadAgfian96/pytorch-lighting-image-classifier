from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT 
from src.net_v2 import ModelClassifier
from lightning.pytorch.callbacks import Callback
from clearml import Task
import torch 
import torchmetrics.functional as fm

class CallbackClearML(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.task:Task = Task.current_task()
        self.logger = self.task.get_logger()
        
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")

    def on_train_batch_end(self, trainer:pl.Trainer, pl_module:ModelClassifier, outputs, batch, batch_idx):
        section = "train"
        self.__generate_report_step_end(
            outputs=outputs, section=section, pl_module=pl_module)

    def on_validation_batch_end(self, trainer:pl.Trainer, pl_module:ModelClassifier, outputs, batch, batch_idx):
        section = "val"
        self.__generate_report_step_end(
            outputs=outputs, section=section, pl_module=pl_module)

    def on_train_epoch_end(self, trainer:pl.Trainer, pl_module:ModelClassifier):
        losses, preds, labels = pl_module.output_train_step.get()
        args = {
            "trainer": trainer, 
            "pl_module": pl_module, 
            "losses": losses, 
            "preds": preds, 
            "labels": labels,
        }
        self.__generate_report_epoch_end(section="train", **args)        
        pl_module.output_train_step.clear() # free up the memory
    
    def on_validation_epoch_end(self, trainer:pl.Trainer, pl_module:ModelClassifier):
        losses, preds, labels = pl_module.output_val_step.get()
        args = {
            "trainer": trainer, 
            "pl_module": pl_module, 
            "losses": losses, 
            "preds": preds, 
            "labels": labels,
        }
        self.__generate_report_epoch_end(section="val", **args)
        pl_module.output_val_step.clear()

    def on_test_epoch_end(self, trainer:pl.Trainer, pl_module:ModelClassifier):
        losses, preds, labels = pl_module.output_test_step.get()
        args = {
            "trainer": trainer, 
            "pl_module": pl_module, 
            "losses": losses, 
            "preds": preds, 
            "labels": labels,
        }
        self.__generate_report_epoch_end(section="test", **args)
        pl_module.output_test_step.clear()

    def __generate_report_step_end(
            self, outputs, section, pl_module:ModelClassifier
        ):
        if type(outputs) is torch.Tensor:
            loss = outputs
        else:
            loss = outputs.get("loss", 0)
        pl_module.log(f"{section}_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.logger.report_scalar(
            title="Loss Step", 
            series=f"loss_{section}", 
            value=loss, 
            iteration=pl_module.global_step
        )

    def __generate_report_epoch_end(self, 
            trainer:pl.Trainer, pl_module:ModelClassifier, 
            losses, preds, labels,
            section:str="train"
        ):
        loss = torch.stack([x for x in losses]).mean()
        
        args_metrics = {
            "preds": torch.cat([x for x in preds]),
            "target": torch.cat([x for x in labels]),
            "task": "multiclass",
            "num_classes": pl_module.d_data.num_classes
        }
        
        # Calculate Metrics
        acc = fm.accuracy(**args_metrics)
        f1 = fm.f1_score(**args_metrics)
        precision = fm.precision(**args_metrics)
        recall = fm.recall(**args_metrics)

        # PyTorch Lightning Logging
        pl_module.log_dict({
            f"{section}_loss": loss,
            f"{section}_acc": acc,
        }, on_epoch=True, prog_bar=True)

        # ClearML Logging
        args_cml ={
            "iteration" : trainer.current_epoch
        }
        self.logger.report_scalar(title="Loss", series=f"loss_{section}", value=loss, **args_cml)
        self.logger.report_scalar(title="Performance", series=f"acc_{section}", value=acc, **args_cml)
        self.logger.report_scalar(title="Performance", series=f"f1_{section}", value=f1, **args_cml)
        self.logger.report_scalar(title="Performance", series=f"precision_{section}", value=precision, **args_cml)
        self.logger.report_scalar(title="Performance", series=f"recall_{section}", value=recall, **args_cml)

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")
