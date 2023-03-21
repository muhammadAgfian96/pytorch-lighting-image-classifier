import timm
from timm.optim import (
    create_optimizer_v2,
    optimizer_kwargs
)
import torch
import pandas as pd
import pytorch_lightning as pl
import torch.optim as optim

from config.default import TrainingConfig
from torchmetrics import (
    Accuracy, 
    ConfusionMatrix, 
    ROC, AUROC
    # AUROC, AUC, ROC, 
    # F1Score, Precision, Recall,
    # PrecisionRecallCurve
)
import matplotlib.pyplot as plt

from torchmetrics.functional import f1_score
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler, PlateauLRScheduler, StepLRScheduler, TanhLRScheduler
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from types import SimpleNamespace
from clearml import Task
import plotly.graph_objects as go
import plotly.express as px
from rich import print
from dataclasses import asdict


def get_lr_scheduler_config(optimizer: torch.optim.Optimizer, 
    LR_SCHEDULER='step', LR_STEP_SIZE=7, LR_DECAY_RATE=0.1, LR_STEP_MILESTONES=[10, 15] ) -> dict:

    if LR_SCHEDULER == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=LR_STEP_SIZE, gamma=LR_DECAY_RATE
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    elif LR_SCHEDULER == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=LR_STEP_MILESTONES, gamma=LR_DECAY_RATE
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    elif LR_SCHEDULER == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=4, threshold=0.0001,
            min_lr=0.000001,
            verbose=True
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'monitor': 'train_loss',
            'interval': 'epoch',
            'frequency': 1,
        }
    else:
        raise NotImplementedError

    return lr_scheduler_config, scheduler

class Classifier(pl.LightningModule):
    def __init__(self, conf:TrainingConfig):
        super().__init__()
        self.conf = conf
        self.model = timm.create_model(
            self.conf.net.architecture,
            pretrained=self.conf.net.checkpoint_model,
            num_classes=self.conf.net.num_class,
            drop_rate=self.conf.net.dropout,
        )
        
        self.classes_name = self.conf.data.category

        self.train_loss = torch.nn.CrossEntropyLoss()
        self.val_loss = torch.nn.CrossEntropyLoss()
        self.test_loss = torch.nn.CrossEntropyLoss()
        
        num_class = self.conf.net.num_class
        # self.task_accuracy = 'multiclass' if num_class > 2 else 'binary'
        self.task_accuracy = 'multiclass' 
        self.train_acc = Accuracy(task=self.task_accuracy, num_classes=num_class)
        self.val_acc = Accuracy(task=self.task_accuracy, num_classes=num_class)
        self.test_acc = Accuracy(task=self.task_accuracy, num_classes=num_class)
        self.roc = ROC(task=self.task_accuracy, num_classes=num_class)
        self.auroc = AUROC(task=self.task_accuracy, num_classes=num_class)
        self.cm = ConfusionMatrix(task=self.task_accuracy, num_classes=num_class)

        self.learning_rate = self.conf.hyp.base_learning_rate
        d_hyp = asdict(self.conf.hyp)
        self.save_hyperparameters({
            'net': {
                'architecture': self.conf.net.architecture,
                'dropout': self.conf.net.dropout,
                'num_class': len(self.classes_name),
                'labels': self.classes_name,
            },
            'preprocessing': {
                'input_size': self.conf.data.input_size,
                'mean': self.conf.data.mean,
                'std': self.conf.data.std,
            },
            'hyperparameters': d_hyp
        })

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)
    
    def configure_optimizers(self):
        opt_d = {
            'Adam': optim.Adam(self.model.parameters(), lr=self.learning_rate),
            'SGD': optim.SGD(self.model.parameters(), lr=self.learning_rate),
            'AdamW': optim.AdamW(self.model.parameters(), lr=self.learning_rate),
            'RMSprop': optim.RMSprop(self.model.parameters(), lr=self.learning_rate),
            'Nadam': optim.NAdam(self.model.parameters(), lr=self.learning_rate),
            'Adadelta': optim.Adadelta(self.model.parameters(), lr=self.learning_rate),
            'Adagrad': optim.Adagrad(self.model.parameters(), lr=self.learning_rate),
            'Adamax': optim.Adamax(self.model.parameters(), lr=self.learning_rate),
        }

        optimizer = opt_d.get(self.conf.hyp.opt_name, optim.Adam(self.model.parameters(), lr=self.conf.hyp.base_learning_rate))
        
        lr_scheduler_config, scheduler = get_lr_scheduler_config(
            optimizer, 
            LR_SCHEDULER=self.conf.hyp.lr_scheduler,
            LR_DECAY_RATE=self.conf.hyp.lr_decay_rate,
            LR_STEP_SIZE=self.conf.hyp.lr_step_size
        )
        return  [optimizer], [lr_scheduler_config]
           
    
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        # self.__visualize_augmentations(imgs)

        preds = self(imgs)
        _, pred = preds.max(1)
        # loss = F.cross_entropy(preds, labels)
        loss = self.train_loss(preds, labels)
        acc = self.train_acc(pred, labels)

        # print(preds[0:5], labels[0:5])
        self.log('train_acc_step', acc)
        self.log('train_loss_step', loss)
        
        Task.current_task().get_logger().report_scalar(title='Accuracy Step', series='Train', value=acc, iteration=self.global_step)
        Task.current_task().get_logger().report_scalar(title='Loss Step', series='Train', value=loss, iteration=self.global_step)

        return {"preds": preds, "labels": labels, "loss": loss, "acc": acc}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outs):
        # compile log
        labels_epoch, preds_epoch, loss_epoch, acc_epoch = self.__get_metrics_epoch(outs)
        self.__send_logger_clearml(labels_epoch, preds_epoch, loss_epoch, acc_epoch, section='Train')

        self.log('train_acc', acc_epoch)
        self.log('train_loss', loss_epoch)
        if self.current_epoch == self.conf.hyp.epoch -1:
            Task.current_task().get_logger().report_single_value('train_acc', acc_epoch)
            Task.current_task().get_logger().report_single_value('train_loss', loss_epoch)

    def validation_step(self, batch, batch_idx):
        imgs, y = batch
        y_hat = self(imgs)
        _, pred = y_hat.max(1)

        loss = self.val_loss(y_hat, y)
        acc = self.val_acc(pred, y)
        self.log('val_acc_step', acc)
        self.log('val_loss_step', loss)

        Task.current_task().get_logger().report_scalar(title='Accuracy Step', series='Validation', value=acc, iteration=self.global_step)
        Task.current_task().get_logger().report_scalar(title='Loss Step', series='Validation', value=loss, iteration=self.global_step)
        return {"preds": y_hat, "labels": y, "loss": loss, "acc": acc}

    def validation_step_end(self, validation_step_outputs):
        return validation_step_outputs
    
    def validation_epoch_end(self, outputs):
        # compile log
        labels_epoch, preds_epoch, loss_epoch, acc_epoch = self.__get_metrics_epoch(outputs)
        self.__send_logger_clearml(labels_epoch, preds_epoch, loss_epoch, acc_epoch, section='Validation')

        self.log('val_acc', acc_epoch)
        self.log('val_loss', loss_epoch)
        if self.current_epoch == self.conf.hyp.epoch -1:
            Task.current_task().get_logger().report_single_value('val_acc', acc_epoch)
            Task.current_task().get_logger().report_single_value('val_loss', loss_epoch)


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
        labels_epoch, preds_epoch, loss_epoch, acc_epoch = self.__get_metrics_epoch(outputs)        
        self.__send_logger_clearml(labels_epoch, preds_epoch, loss_epoch, acc_epoch, section='Test')
        self.log('acc_test', acc_epoch)
        self.log('loss_test', loss_epoch)
        Task.current_task().get_logger().report_single_value('test_acc', acc_epoch)
        Task.current_task().get_logger().report_single_value('test_loss', loss_epoch)


    # generate metrics/plots
    def __confusion_matrix(self, preds, labels):
        # print(preds, labels)

        df_cm = pd.DataFrame(
            self.cm(preds, labels).cpu().numpy(), 
            index=[lbl+'_gt' for lbl in self.classes_name], 
            columns=[lbl+'_pd' for lbl in self.classes_name]
        )
        fig_cm_val = px.imshow(df_cm, text_auto=True, color_continuous_scale=px.colors.sequential.Blues)
        return fig_cm_val

    def __find_best_f1_score(self, preds, labels):
        threshold = np.arange(0.1, 0.99, 0.025, dtype=np.float16).tolist()
        scores_f1 = [f1_score(task=self.task_accuracy, preds=preds, target=labels, threshold=thresh, num_classes=self.conf.net.num_class, top_k=1).item() for thresh in threshold]
        scores_f1 = torch.tensor(scores_f1)
        idx_score_max = torch.argmax(scores_f1).item()
        best_threshold = threshold[idx_score_max]
        best_score = scores_f1[idx_score_max].item()
        return best_score, best_threshold

    def __roc_plot(self, preds_logits, labels):
        self.roc.update(preds_logits, labels)
        self.auroc.update(preds_logits, labels)
                # Get the false positive rate, true positive rate, and threshold for class 1
        result_roc = self.roc.compute()
        result_auc = self.auroc.compute()
        print('result_roc', result_roc)
        print('result_auc', result_auc)
        # Create a plotly graph of the ROC curve and the AUROC curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc_fpr, y=roc_tpr, name='ROC Curve'))
        fig.add_trace(go.Scatter(x=auroc_fpr[1], y=auroc_tpr[1], name='AUROC Curve'))
        fig.update_layout(title='ROC and AUROC Curves', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')

        return fig

    def __table_f1_prec_rec_sup(self, preds, labels):
        probs = torch.softmax(preds, dim=-1)
        _, preds_top1 = torch.max(probs, dim=-1)
        # pred = preds.max(1)[1].detach().cpu().numpy().tolist()
        label = labels.detach().cpu().numpy().tolist()

        d_map = {idx:lbl for idx, lbl in enumerate(self.classes_name)}
        ls_pred = [d_map[p] for p in preds_top1.cpu().tolist()]
        ls_label = [d_map[l] for l in label]

        clsifier_report = precision_recall_fscore_support(
            y_true=ls_label, 
            y_pred=ls_pred, 
            labels=self.classes_name,
            zero_division=1.0)

        d_precision_recall_fbeta_support = {
            'class': self.classes_name,
            'precision': clsifier_report[0],
            'recall': clsifier_report[1],
            'f1_score': clsifier_report[2],
            'count_data': clsifier_report[3]
        }
        df_precision_recall_fbeta_support = pd.DataFrame.from_dict(d_precision_recall_fbeta_support)
        return df_precision_recall_fbeta_support

    def __get_metrics_epoch(self, outs):
        labels_epoch = torch.cat([x["labels"] for x in outs])
        preds_epoch = torch.cat([x["preds"] for x in outs])
        loss_epoch = torch.stack([x["loss"] for x in outs]).mean()
        acc_epoch = torch.stack([x["acc"] for x in outs]).mean()
        return labels_epoch, preds_epoch, loss_epoch, acc_epoch

    def __send_logger_clearml(self, labels_epoch, preds_epoch, loss_epoch, acc_epoch, section):
        # Take the highest probability as the predicted class
        # Convert new_preds to a numpy array
        # probs = torch.softmax(preds_epoch, dim=-1)
        # _, preds_top1 = torch.max(probs, dim=-1)

        fig_cm_val = self.__confusion_matrix(preds_epoch, labels_epoch)
        best_score_f1, best_threshold_f1 = self.__find_best_f1_score(preds_epoch, labels_epoch)
        # fig_roc = self.__roc_plot(preds_epoch, labels_epoch)
        df_table_support = self.__table_f1_prec_rec_sup(preds_epoch, labels_epoch)
        table = pd.DataFrame.from_dict({'Threshold': [best_threshold_f1], 'F1 Score': [best_score_f1]})


        # fig_cm_val.update_xaxes(side="top")
        if section.lower() == 'test':
            iter_ = self.conf.hyp.epoch - 1

        else:
            iter_ = self.current_epoch
        for param_group in self.optimizers().optimizer.param_groups:
            Task.current_task().get_logger().report_scalar(title='LR', series='Train', value=param_group['lr'], iteration=iter_)
        Task.current_task().get_logger().report_scalar(title='Accuracy', series=section, value=acc_epoch, iteration=iter_)
        Task.current_task().get_logger().report_scalar(title='Loss', series=section, value=loss_epoch, iteration=iter_)
        Task.current_task().get_logger().report_scalar(title='F1 Score', series=section, value=best_score_f1, iteration=iter_)
        Task.current_task().get_logger().report_plotly(title='Confusion Matrix', series=section, figure=fig_cm_val, iteration=iter_)
        # Task.current_task().get_logger().report_plotly(title='ROC & AUC', series=section, figure=fig_roc, iteration=iter_)
        Task.current_task().get_logger().report_table(title='Tables', series=f'precision_recall_fscore_support ({section})', table_plot=df_table_support, iteration=iter_)
        Task.current_task().get_logger().report_table(title='Tables', series=f'f1_score ({section})', table_plot=table, iteration=iter_)