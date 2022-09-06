import timm
from timm.optim import (
    create_optimizer_v2,
    optimizer_kwargs
)
from types import SimpleNamespace
import torch
import pytorch_lightning as pl
from clearml import StorageManager, Dataset
from config.default import TrainingConfig
from torchmetrics import Accuracy

class ModelClassifier(pl.LightningModule):
    def __init__(self, conf:TrainingConfig):
        super().__init__()
        self.conf = conf
        self.model = timm.create_model(
            self.conf.net.architecture,
            pretrained=self.conf.net.checkpoint_model,
            num_classes=self.conf.net.num_class
        )
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_loss = torch.nn.CrossEntropyLoss()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
               

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
        preds = self(imgs)
        _, pred = preds.max(1)
        loss = self.train_loss(preds, labels)
        acc = self.train_acc(pred, labels)
        self.log_dict({'acc_train': acc, 'loss_train':loss})
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        _, pred = preds.max(1)
        loss = self.val_loss(preds, labels)
        acc = self.val_acc(pred, labels)
        self.log('acc_val', acc)
        self.log('loss_val', loss)
        return loss

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        _, pred = preds.max(1)
        acc = self.test_acc(pred, labels)
        self.log('acc_train', acc)
