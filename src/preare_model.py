import timm
from timm.optim import (
    create_optimizer_v2,
    optimizer_kwargs
)
import torch
import pandas as pd
import pytorch_lightning as pl
import plotly.express as px

from config.default import TrainingConfig
from torchmetrics import Accuracy, ConfusionMatrix, PrecisionRecallCurve, ROC, F1Score, Precision, Recall
from types import SimpleNamespace
from clearml import Task


class ModelClassifier(pl.LightningModule):
    def __init__(self, conf:TrainingConfig):
        super().__init__()
        self.conf = conf
        self.model = timm.create_model(
            self.conf.net.architecture,
            pretrained=self.conf.net.checkpoint_model,
            num_classes=self.conf.net.num_class
        )
        
        self.classes_name = self.conf.data.category

        self.train_loss = torch.nn.CrossEntropyLoss()
        self.val_loss = torch.nn.CrossEntropyLoss()
        self.test_loss = torch.nn.CrossEntropyLoss()
        
        self.train_acc = Accuracy(num_classes=self.conf.net.num_class)
        self.val_acc = Accuracy(num_classes=self.conf.net.num_class)
        self.test_acc = Accuracy(num_classes=self.conf.net.num_class)
        
        self.train_cm = ConfusionMatrix(num_classes=self.conf.net.num_class)
        self.val_cm = ConfusionMatrix(num_classes=self.conf.net.num_class)
        self.test_cm = ConfusionMatrix(num_classes=self.conf.net.num_class)

        self.save_hyperparameters({
            'net': conf.net,
            'data': conf.data,
            'hyp': conf.hyp
        })

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)
    
    def configure_optimizers(self):
        args_optim = SimpleNamespace()
        args_optim.weight_decay = self.conf.hyp.opt_weight_decay
        args_optim.lr = self.conf.hyp.learning_rate
        args_optim.momentum = self.conf.hyp.opt_momentum 
        args_optim.opt =  self.conf.hyp.opt_name
        
        opt = create_optimizer_v2(
            self.model,
            **optimizer_kwargs(cfg=args_optim)   
        )
        
        return opt
    
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        # print('\nlen_labels_step:', len(labels))
        preds = self(imgs)
        _, pred = preds.max(1)
        loss = self.train_loss(preds, labels)
        acc = self.train_acc(preds, labels)
        self.log('train_acc_step', acc)
        self.log('train_loss_step', loss)

        Task.current_task().get_logger().report_scalar(title='Accuracy Step', series='Train', value=acc, iteration=self.global_step)
        Task.current_task().get_logger().report_scalar(title='Loss Step', series='Train', value=loss, iteration=self.global_step)

        return {"preds": preds, "labels": labels, "loss": loss, "acc": acc}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outs):
        # compile log
        labels_epoch = torch.cat([x["labels"] for x in outs])
        preds_epoch = torch.cat([x["preds"] for x in outs])
        loss_epoch = torch.stack([x["loss"] for x in outs]).mean()
        acc_epoch = torch.stack([x["acc"] for x in outs]).mean()

        df_cm = pd.DataFrame(
            self.train_cm(preds_epoch, labels_epoch).cpu().numpy(), 
            index=[lbl+'_gt' for lbl in self.classes_name], 
            columns=[lbl+'_pd' for lbl in self.classes_name]
        )

        df_cm = pd.DataFrame(self.val_cm(preds_epoch, labels_epoch).cpu().numpy(), index=[lbl+'_gt' for lbl in self.classes_name], columns=[lbl+'_pd' for lbl in self.classes_name])
        fig_cm_val = px.imshow(df_cm, text_auto=True, color_continuous_scale=px.colors.sequential.Blues)
        
        # fig_cm_val.update_xaxes(side="top")
        Task.current_task().get_logger().report_plotly(title='Confusion Matrix', series='Training', figure=fig_cm_val, iteration=self.current_epoch)
        Task.current_task().get_logger().report_scalar(title='Accuracy', series='Train', value=acc_epoch, iteration=self.current_epoch)
        Task.current_task().get_logger().report_scalar(title='Loss', series='Train', value=loss_epoch, iteration=self.current_epoch)

        self.log('train_acc', acc_epoch)
        self.log('train_loss', loss_epoch)

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        loss = self.val_loss(preds, labels)
        acc = self.val_acc(preds, labels)
        self.log('val_acc_step', acc)
        self.log('val_loss_step', loss)

        Task.current_task().get_logger().report_scalar(title='Accuracy Step', series='Validation', value=acc, iteration=self.global_step)
        Task.current_task().get_logger().report_scalar(title='Loss Step', series='Validation', value=loss, iteration=self.global_step)

        return {"preds": preds, "labels": labels, "loss": loss, "acc": acc}

    def validation_step_end(self, validation_step_outputs):
        return validation_step_outputs
    
    def validation_epoch_end(self, outputs):
        # compile log
        labels_epoch = torch.cat([x["labels"] for x in outputs])
        preds_epoch = torch.cat([x["preds"] for x in outputs])
        loss_epoch = torch.stack([x["loss"] for x in outputs]).mean()
        acc_epoch = torch.stack([x["acc"] for x in outputs]).mean()
        
        df_cm = pd.DataFrame(self.val_cm(preds_epoch, labels_epoch).cpu().numpy(), index=[lbl+'_gt' for lbl in self.classes_name], columns=[lbl+'_pd' for lbl in self.classes_name])
        fig_cm_val = px.imshow(df_cm, text_auto=True, color_continuous_scale=px.colors.sequential.Blues)
        
        Task.current_task().get_logger().report_plotly(title='Confusion Matrix', series='Validation', figure=fig_cm_val, iteration=self.current_epoch)
        Task.current_task().get_logger().report_scalar(title='Accuracy', series='Validation', value=acc_epoch, iteration=self.current_epoch)
        Task.current_task().get_logger().report_scalar(title='Loss', series='Validation', value=loss_epoch, iteration=self.current_epoch)

        self.log('val_acc', acc_epoch)
        self.log('val_loss', loss_epoch)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        acc = self.test_acc(preds, labels)
        loss = self.test_loss(preds, labels)
        return {"preds": preds, "labels": labels, "acc": acc, "loss": loss}

    def test_step_end(self, test_step_outputs):
        return test_step_outputs
    
    def test_epoch_end(self, outputs) -> None:
        # get data
        labels_epoch = torch.cat([x["labels"] for x in outputs])
        preds_epoch = torch.cat([x["preds"] for x in outputs])
        acc_epoch = torch.stack([x["acc"] for x in outputs]).mean()
        loss_epoch = torch.stack([x["loss"] for x in outputs]).mean()
        
        # process data
        df_cm = pd.DataFrame(self.test_cm(preds_epoch, labels_epoch).cpu().numpy(), index=[lbl+'_gt' for lbl in self.classes_name], columns=[lbl+'_pd' for lbl in self.classes_name])
        fig_cm_val = px.imshow(df_cm, text_auto=True, color_continuous_scale=px.colors.sequential.Blues)
        # fig_cm_val.update_xaxes(side="top")
        
        # store logs
        Task.current_task().get_logger().report_plotly(title='Confusion Matrix', series='Test', figure=fig_cm_val, iteration=self.current_epoch)
        Task.current_task().get_logger().report_scalar(title='Accuracy', series='Test', value=acc_epoch, iteration=self.current_epoch)
        Task.current_task().get_logger().report_scalar(title='Loss', series='Test', value=loss_epoch, iteration=self.current_epoch)

        self.log('acc_test', acc_epoch)
        self.log('loss_test', loss_epoch)
