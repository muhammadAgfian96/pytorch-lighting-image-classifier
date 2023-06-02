import os
from collections import defaultdict
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytorch_lightning as pl
import timm
import torch
import torch.optim as optim
from clearml import InputModel, Task
from PIL import Image
from rich import print
from sklearn.metrics import precision_recall_fscore_support
from torchmetrics import (  # AUROC, AUC, ROC,; F1Score, Precision, Recall,; PrecisionRecallCurve
    AUROC, ROC, Accuracy, ConfusionMatrix)
from torchmetrics.functional import f1_score

from config.default import TrainingConfig
from config.list_models import list_models as ls_models_library
from utils_roc import generate_plot_one_vs_one, generate_plot_one_vs_rest


def denormalize_image(image, mean, std):
    img_copy = image.copy()
    for i in range(img_copy.shape[2]):
        img_copy[:, :, i] = img_copy[:, :, i] * std[i] + mean[i]
    return img_copy


def get_lr_scheduler_config(
    optimizer: torch.optim.Optimizer,
    LR_SCHEDULER="step",
    LR_STEP_SIZE=7,
    LR_DECAY_RATE=0.1,
    LR_STEP_MILESTONES=[10, 15],
) -> dict:
    if LR_SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=LR_STEP_SIZE, gamma=LR_DECAY_RATE
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
    elif LR_SCHEDULER == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=LR_STEP_MILESTONES, gamma=LR_DECAY_RATE
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
    elif LR_SCHEDULER == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=4,
            threshold=0.0001,
            min_lr=0.000001,
            verbose=True,
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "monitor": "train_loss",
            "interval": "epoch",
            "frequency": 1,
        }
    else:
        raise NotImplementedError

    return lr_scheduler_config, scheduler


class Classifier(pl.LightningModule):
    def __init__(self, conf: TrainingConfig):
        super().__init__()
        self.conf = conf

        if self.conf.net.architecture not in ls_models_library and len(
            self.conf.net.architecture
        ) == len("4d609a6b07ed447bae47af9a6fbfb999"):
            model_old = InputModel(model_id=self.conf.net.architecture)
            self.conf.net.architecture = model_old.config_dict["net"]
            print("dowloding model", self.conf.net.architecture)
            print("old model labels", model_old.labels)
            model_old_path = model_old.get_weights()
            self.conf.net.checkpoint_model = model_old_path

        self.model = timm.create_model(
            self.conf.net.architecture,
            pretrained=self.conf.net.checkpoint_model,
            num_classes=self.conf.net.num_class,
            drop_rate=self.conf.net.dropout,
            # checkpoint_path=(self.conf.net.checkpoint_model),
        )
        if self.conf.net.checkpoint_model is not None:
            # torchscript way
            # loaded_script_model = torch.jit.load(self.conf.net.checkpoint_model)
            # self.model.load_state_dict(loaded_script_model.state_dict())

            # raw way
            checkpoint = torch.load(self.conf.net.checkpoint_model)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            Task.current_task().add_tags("resume")

        Task.current_task().add_tags(conf.net.architecture)
        self.classes_name = self.conf.data.category

        self.train_loss = torch.nn.CrossEntropyLoss()
        self.val_loss = torch.nn.CrossEntropyLoss()
        self.test_loss = torch.nn.CrossEntropyLoss()

        num_class = self.conf.net.num_class
        # self.task_accuracy = 'multiclass' if num_class > 2 else 'binary'
        self.task_accuracy = "multiclass"
        self.train_acc = Accuracy(task=self.task_accuracy, num_classes=num_class)
        self.val_acc = Accuracy(task=self.task_accuracy, num_classes=num_class)
        self.test_acc = Accuracy(task=self.task_accuracy, num_classes=num_class)
        self.roc = ROC(task=self.task_accuracy, num_classes=num_class)
        self.auroc = AUROC(task=self.task_accuracy, num_classes=num_class)
        self.cm = ConfusionMatrix(task=self.task_accuracy, num_classes=num_class)

        self.learning_rate = self.conf.hyp.base_learning_rate
        d_hyp = asdict(self.conf.hyp)
        self.save_hyperparameters(
            {
                "net": {
                    "architecture": self.conf.net.architecture,
                    "dropout": self.conf.net.dropout,
                    "num_class": len(self.classes_name),
                    "labels": self.classes_name,
                },
                "preprocessing": {
                    "input_size": self.conf.data.input_size,
                    "mean": self.conf.data.mean,
                    "std": self.conf.data.std,
                },
                "hyperparameters": d_hyp,
            }
        )

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        opt_d = {
            "Adam": optim.Adam(self.model.parameters(), lr=self.learning_rate),
            "SGD": optim.SGD(self.model.parameters(), lr=self.learning_rate),
            "AdamW": optim.AdamW(self.model.parameters(), lr=self.learning_rate),
            "RMSprop": optim.RMSprop(self.model.parameters(), lr=self.learning_rate),
            "Nadam": optim.NAdam(self.model.parameters(), lr=self.learning_rate),
            "Adadelta": optim.Adadelta(self.model.parameters(), lr=self.learning_rate),
            "Adagrad": optim.Adagrad(self.model.parameters(), lr=self.learning_rate),
            "Adamax": optim.Adamax(self.model.parameters(), lr=self.learning_rate),
        }

        optimizer = opt_d.get(
            self.conf.hyp.opt_name,
            optim.Adam(self.model.parameters(), lr=self.conf.hyp.base_learning_rate),
        )

        lr_scheduler_config, scheduler = get_lr_scheduler_config(
            optimizer,
            LR_SCHEDULER=self.conf.hyp.lr_scheduler,
            LR_DECAY_RATE=self.conf.hyp.lr_decay_rate,
            LR_STEP_SIZE=self.conf.hyp.lr_step_size,
        )
        return [optimizer], [lr_scheduler_config]

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        # self.__visualize_augmentations(imgs)

        preds = self(imgs)
        _, pred = preds.max(1)
        # loss = F.cross_entropy(preds, labels)
        loss = self.train_loss(preds, labels)
        acc = self.train_acc(pred, labels)

        # print(preds[0:5], labels[0:5])
        self.log("train_acc_step", acc)
        self.log("train_loss_step", loss)

        Task.current_task().get_logger().report_scalar(
            title="Accuracy Step",
            series="Train",
            value=acc,
            iteration=self.global_step,
        )
        Task.current_task().get_logger().report_scalar(
            title="Loss Step",
            series="Train",
            value=loss,
            iteration=self.global_step,
        )

        return {"preds": preds, "labels": labels, "loss": loss, "acc": acc}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outs):
        # compile log
        (
            labels_epoch,
            preds_epoch,
            loss_epoch,
            acc_epoch,
        ) = self.__get_metrics_epoch(outs)
        self.__send_logger_clearml(
            labels_epoch, preds_epoch, loss_epoch, acc_epoch, section="Train"
        )

        self.log("train_acc", acc_epoch)
        self.log("train_loss", loss_epoch)
        if self.current_epoch == self.conf.hyp.epoch - 1:
            Task.current_task().get_logger().report_single_value("train_acc", acc_epoch)
            Task.current_task().get_logger().report_single_value(
                "train_loss", loss_epoch
            )

    def validation_step(self, batch, batch_idx):
        imgs, y = batch
        y_hat = self(imgs)
        _, pred = y_hat.max(1)

        loss = self.val_loss(y_hat, y)
        acc = self.val_acc(pred, y)
        self.log("val_acc_step", acc)
        self.log("val_loss_step", loss)

        Task.current_task().get_logger().report_scalar(
            title="Accuracy Step",
            series="Validation",
            value=acc,
            iteration=self.global_step,
        )
        Task.current_task().get_logger().report_scalar(
            title="Loss Step",
            series="Validation",
            value=loss,
            iteration=self.global_step,
        )
        return {
            "preds": y_hat,
            "labels": y,
            "loss": loss,
            "acc": acc,
            "imgs": imgs,
        }

    def validation_step_end(self, validation_step_outputs):
        return validation_step_outputs

    def validation_epoch_end(self, outputs):
        # compile log
        (
            labels_epoch,
            preds_epoch,
            loss_epoch,
            acc_epoch,
        ) = self.__get_metrics_epoch(outputs)
        self.__send_logger_clearml(
            labels_epoch,
            preds_epoch,
            loss_epoch,
            acc_epoch,
            section="Validation",
        )

        self.log("val_acc", acc_epoch)
        self.log("val_loss", loss_epoch)
        if self.current_epoch == self.conf.hyp.epoch - 1:
            Task.current_task().get_logger().report_single_value("val_acc", acc_epoch)
            Task.current_task().get_logger().report_single_value("val_loss", loss_epoch)

        if (
            self.current_epoch % 25 == 0
            or self.current_epoch == self.conf.hyp.epoch - 1
        ):
            self.visualize_images(
                test_outputs=outputs,
                section="validation",
                epoch=self.current_epoch,
                upload_to_clearml=True,
                row_x_col=(5, 5),
                gt_label_limit=50,
            )

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        acc = self.test_acc(preds, labels)
        loss = self.test_loss(preds, labels)

        return {
            "preds": preds,
            "labels": labels,
            "acc": acc,
            "loss": loss,
            "imgs": imgs,
        }

    def test_step_end(self, test_step_outputs):
        return test_step_outputs

    def test_epoch_end(self, outputs) -> None:
        # get data
        (
            labels_epoch,
            preds_epoch,
            loss_epoch,
            acc_epoch,
        ) = self.__get_metrics_epoch(outputs)
        self.__send_logger_clearml(
            labels_epoch, preds_epoch, loss_epoch, acc_epoch, section="Test"
        )
        self.log("acc_test", acc_epoch)
        self.log("loss_test", loss_epoch)
        Task.current_task().get_logger().report_single_value("test_acc", acc_epoch)
        Task.current_task().get_logger().report_single_value("test_loss", loss_epoch)
        self.visualize_images(
            test_outputs=outputs,
            section="test",
            epoch=self.current_epoch,
            upload_to_clearml=True,
            row_x_col=(5, 5),
            gt_label_limit=200,
        )

    # generate metrics/plots
    def __confusion_matrix(self, preds, labels):
        # print(preds, labels)

        df_cm = pd.DataFrame(
            self.cm(preds, labels).cpu().numpy(),
            index=[lbl + "_gt" for lbl in self.classes_name],
            columns=[lbl + "_pd" for lbl in self.classes_name],
        )
        fig_cm_val = px.imshow(
            df_cm,
            text_auto=True,
            color_continuous_scale=px.colors.sequential.Blues,
        )
        return fig_cm_val

    def __find_best_f1_score(self, preds, labels):
        threshold = np.arange(0.1, 0.99, 0.025, dtype=np.float16).tolist()
        scores_f1 = [
            f1_score(
                task=self.task_accuracy,
                preds=preds,
                target=labels,
                threshold=thresh,
                num_classes=self.conf.net.num_class,
                top_k=1,
            ).item()
            for thresh in threshold
        ]
        scores_f1 = torch.tensor(scores_f1)
        idx_score_max = torch.argmax(scores_f1).item()
        best_threshold = threshold[idx_score_max]
        best_score = scores_f1[idx_score_max].item()
        return best_score, best_threshold

    def __roc_plot(self, preds_softmax, labels, section):
        os.makedirs("logger_roc", exist_ok=True)
        gt_truth = [self.classes_name[idx_lbl] for idx_lbl in labels.cpu().tolist()]
        preds_softmax_np = preds_softmax.detach().cpu().numpy()
        params_task_ovr = {
            "title": f"ROC {section}",
            "series": "OneVsRest",
            "iteration": self.current_epoch,
        }
        params_task_ovo = {
            "title": f"ROC {section}",
            "series": "OneVsOne",
            "iteration": self.current_epoch,
        }

        generate_plot_one_vs_rest(
            class_names=self.classes_name,
            gt_labels=gt_truth,
            preds_softmax=preds_softmax_np,
            path_to_save="logger_roc",
            task=Task.current_task(),
            **params_task_ovr,
        )

        generate_plot_one_vs_one(
            class_names=self.classes_name,
            gt_labels=gt_truth,
            preds_softmax=preds_softmax_np,
            task=Task.current_task() if len(self.classes_name) <= 10 else None,
            **params_task_ovo,
        )

    def __table_f1_prec_rec_sup(self, preds, labels):
        probs = torch.softmax(preds, dim=-1)
        _, preds_top1 = torch.max(probs, dim=-1)
        # pred = preds.max(1)[1].detach().cpu().numpy().tolist()
        label = labels.detach().cpu().numpy().tolist()

        d_map = {idx: lbl for idx, lbl in enumerate(self.classes_name)}
        ls_pred = [d_map[p] for p in preds_top1.cpu().tolist()]
        ls_label = [d_map[l] for l in label]

        clsifier_report = precision_recall_fscore_support(
            y_true=ls_label,
            y_pred=ls_pred,
            labels=self.classes_name,
            zero_division=1.0,
        )

        d_precision_recall_fbeta_support = {
            "class": self.classes_name,
            "precision": clsifier_report[0],
            "recall": clsifier_report[1],
            "f1_score": clsifier_report[2],
            "count_data": clsifier_report[3],
        }
        df_precision_recall_fbeta_support = pd.DataFrame.from_dict(
            d_precision_recall_fbeta_support
        )
        return df_precision_recall_fbeta_support

    def __get_metrics_epoch(self, outs):
        labels_epoch = torch.cat([x["labels"] for x in outs])
        preds_epoch = torch.cat([x["preds"] for x in outs])
        loss_epoch = torch.stack([x["loss"] for x in outs]).mean()
        acc_epoch = torch.stack([x["acc"] for x in outs]).mean()
        return labels_epoch, preds_epoch, loss_epoch, acc_epoch

    def __send_logger_clearml(
        self, labels_epoch, preds_epoch, loss_epoch, acc_epoch, section
    ):
        # Take the highest probability as the predicted class
        # Convert new_preds to a numpy array
        probs = torch.softmax(preds_epoch, dim=-1)
        probs_top1, preds_top1 = torch.max(probs, dim=-1)

        fig_cm_val = self.__confusion_matrix(preds_epoch, labels_epoch)
        best_score_f1, best_threshold_f1 = self.__find_best_f1_score(
            preds_epoch, labels_epoch
        )
        df_table_support = self.__table_f1_prec_rec_sup(preds_epoch, labels_epoch)
        table = pd.DataFrame.from_dict(
            {"Threshold": [best_threshold_f1], "F1 Score": [best_score_f1]}
        )

        # fig_cm_val.update_xaxes(side="top")
        if section.lower() == "test":
            iter_ = self.conf.hyp.epoch - 1
        else:
            iter_ = self.current_epoch

        if (
            self.current_epoch == self.conf.hyp.epoch - 1
            and section.lower() == "validation"
        ) or section.lower() == "test":
            try:
                # report ROC if last epoch
                print("report ROC if last epoch")
                self.__roc_plot(probs, labels_epoch, section)
            except Exception as e:
                print(f"Error in ROC plot: {e}")

        for param_group in self.optimizers().optimizer.param_groups:
            Task.current_task().get_logger().report_scalar(
                title="LR",
                series="Train",
                value=param_group["lr"],
                iteration=iter_,
            )

        Task.current_task().get_logger().report_scalar(
            title="Accuracy", series=section, value=acc_epoch, iteration=iter_
        )
        Task.current_task().get_logger().report_scalar(
            title="Loss", series=section, value=loss_epoch, iteration=iter_
        )
        Task.current_task().get_logger().report_scalar(
            title="F1 Score",
            series=f"{section}",
            value=best_score_f1,
            iteration=iter_,
        )
        Task.current_task().get_logger().report_plotly(
            title="Confusion Matrix",
            series=section,
            figure=fig_cm_val,
            iteration=iter_,
        )
        Task.current_task().get_logger().report_table(
            title="Tables",
            series=f"precision_recall_fscore_support ({section})",
            table_plot=df_table_support,
            iteration=iter_,
        )
        Task.current_task().get_logger().report_table(
            title="Tables",
            series=f"f1_score ({section})",
            table_plot=table,
            iteration=iter_,
        )

    def visualize_images(
        self,
        test_outputs,
        epoch,
        section,
        row_x_col=(5, 5),
        upload_to_clearml=True,
        gt_label_limit=50,
    ):
        FOLDER_SAVE = "logger_predict"
        os.makedirs(f"{FOLDER_SAVE}", exist_ok=True)
        n_row, n_col = row_x_col
        img_counter = 0
        mean = self.conf.data.mean
        std = self.conf.data.std
        gt_label_count = defaultdict(int)

        fig, axes = plt.subplots(n_row, n_col, figsize=(15, 15))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        for batch_idx in range(len(test_outputs)):
            batch_images = test_outputs[batch_idx]["imgs"]
            batch_labels = test_outputs[batch_idx]["labels"]
            batch_preds = test_outputs[batch_idx]["preds"]

            for sample_idx in range(len(batch_images)):
                gt_label = batch_labels[sample_idx].item()
                if gt_label_count[gt_label] > gt_label_limit:
                    continue
                gt_label_count[gt_label] += 1

                img = batch_images[sample_idx].permute(1, 2, 0).cpu().numpy()
                img = denormalize_image(img, mean, std)

                pred_label = batch_preds[sample_idx].argmax(dim=-1).item()
                confidence = batch_preds[sample_idx].softmax(dim=-1).max().item()
                conf = round(confidence * 100, 2)

                title_color = "green" if pred_label == gt_label else "red"
                name_gt = self.classes_name[gt_label]
                name_pred = self.classes_name[pred_label]

                row_idx = img_counter // n_col
                col_idx = img_counter % n_col

                if n_row > 1:
                    ax = axes[row_idx, col_idx]
                else:
                    ax = axes[col_idx]

                # add to plot every image
                ax.imshow(img)
                ax.set_title(f"{name_gt} vs {name_pred} ({conf})", color=title_color)
                ax.axis("off")

                img_counter += 1

                # If 25 images are displayed, reset the counter and create a new figure
                # and clear the plot
                if img_counter == n_row * n_col:
                    plt.tight_layout()
                    naming_file = (
                        f"{section.upper()}_b{batch_idx}_s{sample_idx}_e{epoch}"
                    )
                    path_file_predict = f"{FOLDER_SAVE}/{naming_file}.jpg"
                    plt.savefig(path_file_predict)
                    plt.close()
                    if upload_to_clearml:
                        Task.current_task().get_logger().report_image(
                            f"{section.capitalize()}-Predict",
                            naming_file,
                            iteration=epoch,
                            image=Image.open(path_file_predict),
                        )
                    # reset and create again
                    img_counter = 0
                    fig, axes = plt.subplots(n_row, n_col, figsize=(15, 15))
                    fig.subplots_adjust(hspace=0.5, wspace=0.5)

            # Show the remaining images in the last grid, if there are any
            if img_counter > 0:
                for i in range(img_counter, n_row * n_col):
                    row_idx = i // n_col
                    col_idx = i % n_col
                    if n_row > 1:
                        axes[row_idx, col_idx].axis("off")
                    else:
                        axes[col_idx].axis("off")

                plt.tight_layout()
                naming_file = (
                    f"{section.upper()}_b{batch_idx}_s{sample_idx}_e{epoch}_last"
                )
                path_file_predict = f"{FOLDER_SAVE}/{naming_file}.jpg"
                plt.savefig(path_file_predict)
                plt.close()
                if upload_to_clearml:
                    Task.current_task().get_logger().report_image(
                        f"{section.capitalize()}-Predict",
                        naming_file,
                        iteration=epoch,
                        image=Image.open(path_file_predict),
                    )
                img_counter = 0
                fig, axes = plt.subplots(n_row, n_col, figsize=(15, 15))
                fig.subplots_adjust(hspace=0.5, wspace=0.5)
