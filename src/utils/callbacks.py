from collections import defaultdict
import os
from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib import pyplot as plt 
from src.net_v2 import ModelClassifier
from lightning.pytorch.callbacks import Callback
from clearml import Task
import torch 
import torchmetrics.functional as fm
import pandas as pd
import plotly.express as px
from src.utils.utils import denormalize_image, denormalize_imagev2
from PIL import Image
from uuid import uuid4

class CallbackClearML(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.task:Task = Task.current_task()
        self.logger = self.task.get_logger()
    
    def on_train_start(self, trainer, pl_module:ModelClassifier):
        print("Training is started!")
        self.task.add_tags(pl_module.d_model.architecture)
        self.task.add_tags(f"img_sz:{pl_module.d_model.input_size}")

    def on_train_batch_end(self, trainer:pl.Trainer, pl_module:ModelClassifier, outputs, batch, batch_idx):
        section = "train"
        self.__generate_report_step_end(outputs=outputs, section=section, pl_module=pl_module)

    def on_train_epoch_end(self, trainer:pl.Trainer, pl_module:ModelClassifier):
        losses, preds, labels, imgs = pl_module.output_train_step.get()
        args = {
            "trainer": trainer, 
            "pl_module": pl_module, 
            "losses": losses, 
            "preds": preds, 
            "labels": labels,
        }
        self.__generate_report_epoch_end(section="train", **args)
        pl_module.output_train_step.clear() # free up the memory

    def on_validation_batch_end(self, trainer:pl.Trainer, pl_module:ModelClassifier, outputs, batch, batch_idx):
        section = "val"
        self.__generate_report_step_end(outputs=outputs, section=section, pl_module=pl_module)

    def on_validation_epoch_end(self, trainer:pl.Trainer, pl_module:ModelClassifier):
        losses, preds, labels, imgs = pl_module.output_val_step.get()
        args = {
            "trainer": trainer, 
            "pl_module": pl_module, 
            "losses": losses, 
            "preds": preds, 
            "labels": labels,
        }
        self.__generate_report_epoch_end(section="val", **args)
        pl_module.output_val_step.clear()

    # def on_test_batch_end(self, trainer:pl.Trainer, pl_module:ModelClassifier, outputs, batch, batch_idx):
        # _, preds, labels, imgs = pl_module.output_test_step.get()
        # self.__visualize_images(
        #     imgs=imgs, labels=labels, preds=preds,
        #     epoch=pl_module.current_epoch, section="test",
        #     pl_module= pl_module
        # )
        

    def on_test_epoch_end(self, trainer:pl.Trainer, pl_module:ModelClassifier):
        losses, preds, labels, imgs = pl_module.output_test_step.get()
        args = {
            "trainer": trainer, 
            "pl_module": pl_module, 
            "losses": losses, 
            "preds": preds, 
            "labels": labels,
        }
        self.__generate_report_epoch_end(section="test", **args)
        self.__visualize_images(
            imgs=imgs, labels=labels, preds=preds,
            epoch=pl_module.current_epoch, section="test",
            pl_module=pl_module
        )
        pl_module.output_test_step.clear()

    def on_train_end(self, trainer:pl.Trainer, pl_module:ModelClassifier):
        # memory_consumption = pl_module.final_vram_memory - pl_module.initial_vram_memory
        # self.logger.report_single_value("model vram (MB)", memory_consumption)
        self.logger.report_single_value("train_acc", trainer.callback_metrics["train_acc"])

    def on_validation_end(self, trainer, pl_module):
        self.logger.report_single_value("val_acc", trainer.callback_metrics["val_acc"])

    def on_test_end(self, trainer: pl.Trainer, pl_module:ModelClassifier):
        self.logger.report_single_value("test_acc", trainer.callback_metrics["test_acc"])
 

    # <==================== functions ========================>
    def __generate_report_step_end(self, outputs, section, pl_module:ModelClassifier):
        if type(outputs) is torch.Tensor:
            # for validations
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
        tensor_cm = fm.confusion_matrix(
            **args_metrics, threshold=0.6
        )
        # PyTorch Lightning Logging
        pl_module.log_dict({
            f"{section}_loss": loss,
            f"{section}_acc": acc,
        }, on_epoch=True, prog_bar=True)

        # ClearML Logging
        args_cml ={
            "iteration" : trainer.current_epoch
        }
        if section == "test": args_cml["iteration"] -= 1
        
        self.logger.report_scalar(title="Loss", series=f"loss_{section}", value=loss, **args_cml)
        self.logger.report_scalar(title="Accuracy", series=f"acc_{section}", value=acc, **args_cml)
        self.logger.report_scalar(title="F1 Score", series=f"f1_{section}", value=f1, **args_cml)
        self.logger.report_scalar(title="Precision", series=f"precision_{section}", value=precision, **args_cml)
        self.logger.report_scalar(title="Recall", series=f"recall_{section}", value=recall, **args_cml)

        if section == "train":
            self.logger.report_scalar("Learning Rate", "lr", pl_module.learning_rate, pl_module.current_epoch)
        
        # 

        df_cm = pd.DataFrame(
            tensor_cm.cpu().numpy(),
            index=[lbl + "_gt" for lbl in pl_module.d_data.classes],
            columns=[lbl + "_pd" for lbl in pl_module.d_data.classes],
        )
        fig_cm = px.imshow(
            df_cm,
            text_auto=True,
            color_continuous_scale=px.colors.sequential.Blues,
        )
        # self.logger.report_confusion_matrix(
        #     title="Confusion Matrix Logger", 
        #     series=f"{section.capitalize()}", 
        #     matrix=tensor_cm.cpu().numpy(), 
        #     iteration=pl_module.current_epoch,
        #     xaxis="Predicted",
        #     yaxis="Ground Truth", 
        #     xlabels=pl_module.d_data.classes,
        #     ylabels=pl_module.d_data.classes,
        # )
        self.logger.report_plotly(title="Confusion Matrix", series=f"{section.capitalize()}", iteration=pl_module.current_epoch, figure=fig_cm) 

    
    def __visualize_images(self,
        imgs, labels, preds,
        epoch,
        section,
        row_x_col=(5, 5),
        upload_to_clearml=True,
        gt_label_limit=150,
        pl_module:ModelClassifier=None
    ):
        FOLDER_SAVE = "logger_predict"
        os.makedirs(f"{FOLDER_SAVE}", exist_ok=True)
        n_row, n_col = row_x_col
        img_counter = 0
        mean = pl_module.d_data.mean
        std = pl_module.d_data.std
        gt_label_count = defaultdict(int)

        fig, axes = plt.subplots(n_row, n_col, figsize=(15, 15))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        args_metrics = {
            "preds": torch.cat([x for x in preds]),
            "target": torch.cat([x for x in labels]),
            "images": torch.cat([x for x in imgs]),
        }
        predictions, targets, images = args_metrics["preds"], args_metrics["target"], args_metrics["images"]

        for predict, target, image in zip(predictions, targets, images):
            gt_label = target.item()
            if gt_label_count[gt_label] > gt_label_limit:
                continue
            gt_label_count[gt_label] += 1

            img = image.permute(1, 2, 0).cpu().numpy()
            img = denormalize_imagev2(img, mean, std)

            pred_label = predict.argmax(dim=-1).item()
            confidence = predict.softmax(dim=-1).max().item()
            pred_prob = round(confidence * 100, 2)

            title_color = "green" if pred_label == gt_label else "red"

            txt_label = pl_module.d_data.classes[gt_label].capitalize()
            txt_pred = pl_module.d_data.classes[pred_label].capitalize()
            
            title_img = f"Truth: {txt_label}\nPred: {txt_pred} {pred_prob}%"
            axes[img_counter // n_col, img_counter % n_col].imshow(img)
            axes[img_counter // n_col, img_counter % n_col].set_title(title_img, color=title_color)
            axes[img_counter // n_col, img_counter % n_col].axis("off")
            img_counter += 1

            if img_counter == n_row * n_col:
                naming_file, path_file_predict = self.__save_plt_img(section, FOLDER_SAVE, img_counter, gt_label)

                if upload_to_clearml:
                    self.logger.report_image(
                        f"new-{section.capitalize()}-Predict",
                        naming_file,
                        iteration=epoch,
                        image=Image.open(path_file_predict),
                    )
                # reset and create again
                img_counter = 0
                fig, axes = plt.subplots(n_row, n_col, figsize=(15, 15))
                fig.subplots_adjust(hspace=0.5, wspace=0.5)
                print("success logging")

        if img_counter > 0:
            for i in range(img_counter, n_row * n_col):
                row_idx = i // n_col
                col_idx = i % n_col
                if n_row > 1:
                    axes[row_idx, col_idx].axis("off")
                else:
                    axes[col_idx].axis("off")

            naming_file, path_file_predict = self.__save_plt_img(section, FOLDER_SAVE, img_counter, gt_label)

            if upload_to_clearml:
                self.logger.report_image(
                    f"new-{section.capitalize()}-Predict",
                    naming_file,
                    iteration=epoch,
                    image=Image.open(path_file_predict),
                )
            img_counter = 0
            fig, axes = plt.subplots(n_row, n_col, figsize=(15, 15))
            fig.subplots_adjust(hspace=0.5, wspace=0.5)

    def __save_plt_img(self, section, FOLDER_SAVE, img_counter, gt_label):
        plt.tight_layout()
        naming_file = (
                    f"{section.upper()}_{gt_label}_{img_counter}_{uuid4().hex}"
                )
        path_file_predict = f"{FOLDER_SAVE}/{naming_file}.jpg"
        plt.savefig(path_file_predict)
        plt.close()
        return naming_file,path_file_predict