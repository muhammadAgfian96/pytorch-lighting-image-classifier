import timm
from timm.optim import (
    create_optimizer_v2,
    optimizer_kwargs
)
import torch
import pandas as pd
import pytorch_lightning as pl

from config.default import TrainingConfig
from torchmetrics import (
    Accuracy, 
    ConfusionMatrix, 
    AUROC, AUC, ROC, 
    F1Score, Precision, Recall,
    PrecisionRecallCurve)

from torchmetrics.functional import (
    f1_score, accuracy, precision, recall,
    precision_recall, precision_recall_curve,
    auc, auroc, roc, confusion_matrix
)

from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from types import SimpleNamespace
from clearml import Task
import plotly.graph_objects as go
import plotly.express as px
from rich import print


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
        
        self.cm = ConfusionMatrix(num_classes=self.conf.net.num_class)


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
        labels_epoch, preds_epoch, loss_epoch, acc_epoch = self.__get_metrics_epoch(outs)
        self.__send_logger_clearml(labels_epoch, preds_epoch, loss_epoch, acc_epoch, section='Train')

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
        labels_epoch, preds_epoch, loss_epoch, acc_epoch = self.__get_metrics_epoch(outputs)
        self.__send_logger_clearml(labels_epoch, preds_epoch, loss_epoch, acc_epoch, section='Validation')

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
        labels_epoch, preds_epoch, loss_epoch, acc_epoch = self.__get_metrics_epoch(outputs)        
        self.__send_logger_clearml(labels_epoch, preds_epoch, loss_epoch, acc_epoch, section='Test')
        self.log('acc_test', acc_epoch)
        self.log('loss_test', loss_epoch)

    # generate metrics/plots
    def __confusion_matrix(self, preds, labels):
        df_cm = pd.DataFrame(
            self.cm(preds, labels).cpu().numpy(), 
            index=[lbl+'_gt' for lbl in self.classes_name], 
            columns=[lbl+'_pd' for lbl in self.classes_name]
        )
        fig_cm_val = px.imshow(df_cm, text_auto=True, color_continuous_scale=px.colors.sequential.Blues)
        return fig_cm_val

    def __find_best_f1_score(self, preds, labels):
        threshold = np.arange(0.1, 0.99, 0.025, dtype=np.float16).tolist()
        scores_f1 = [f1_score(preds=preds, target=labels, threshold=thresh, num_classes=self.conf.net.num_class, top_k=1).item() for thresh in threshold]
        scores_f1 = torch.tensor(scores_f1)
        idx_score_max = torch.argmax(scores_f1).item()
        print('threshold', threshold)
        print('scores_f1', scores_f1)
        best_threshold = threshold[idx_score_max]
        best_score = scores_f1[idx_score_max].item()
        return best_score, best_threshold

    def __roc_plot(self, preds, labels):
        fpr_all, tpr_all, thresholds_all = roc(preds, labels, self.conf.net.num_class)
        d = {}
        for idx, lbl in enumerate(self.classes_name):
            d[lbl] = {}
            d[lbl]['fpr'] = fpr_all[idx]
            d[lbl]['tpr'] = tpr_all[idx]
            d[lbl]['thresh'] = thresholds_all[idx]
            
        # graphic
        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        for lbl, d_val in d.items():
            fpr = d_val['fpr']
            tpr = d_val['tpr']
            thr = d_val['thresh']
            auc_score = auc(fpr, tpr, self.conf.net.num_class)

            name = f"{lbl} (AUC={auc_score:.3f})"
            fig.add_trace(go.Scatter(x=fpr.cpu().numpy(), y=tpr.cpu().numpy(), name=name, mode='lines'))

        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
        )
        return fig

    def __table_f1_prec_rec_sup(self, preds, labels):
        pred = preds.max(1)[1].detach().cpu().numpy().tolist()
        label = labels.detach().cpu().numpy().tolist()

        d_map = {idx:lbl for idx, lbl in enumerate(self.classes_name)}
        ls_pred = [d_map[p] for p in pred]
        ls_label = [d_map[l] for l in label]

        clsifier_report = precision_recall_fscore_support(
            y_true=ls_label, 
            y_pred=ls_pred, 
            labels=self.classes_name)

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
        return labels_epoch,preds_epoch,loss_epoch,acc_epoch

    def __send_logger_clearml(self, labels_epoch, preds_epoch, loss_epoch, acc_epoch, section):
        fig_cm_val = self.__confusion_matrix(preds_epoch, labels_epoch)
        best_score_f1, best_threshold_f1 = self.__find_best_f1_score(preds_epoch, labels_epoch)
        fig_roc = self.__roc_plot(preds_epoch, labels_epoch)
        df_table_support = self.__table_f1_prec_rec_sup(preds_epoch, labels_epoch)
        table = pd.DataFrame.from_dict({'Threshold': [best_threshold_f1], 'F1 Score': [best_score_f1]})

        # fig_cm_val.update_xaxes(side="top")
        Task.current_task().get_logger().report_plotly(title='Confusion Matrix', series=section, figure=fig_cm_val, iteration=self.current_epoch)
        Task.current_task().get_logger().report_scalar(title='Accuracy', series=section, value=acc_epoch, iteration=self.current_epoch)
        Task.current_task().get_logger().report_scalar(title='Loss', series=section, value=loss_epoch, iteration=self.current_epoch)
        Task.current_task().get_logger().report_plotly(title='ROC & AUC', series=section, figure=fig_roc, iteration=self.current_epoch)
        Task.current_task().get_logger().report_scalar(title='F1 Score', series=section, value=best_score_f1, iteration=self.current_epoch)
        Task.current_task().get_logger().report_table(title='Tables', series=f'precision_recall_fscore_support ({section})', table_plot=df_table_support, iteration=self.current_epoch)
        Task.current_task().get_logger().report_table(title='Tables', series=f'f1_score ({section})', table_plot=table, iteration=self.current_epoch)

