import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
# from pytorch_lightning.metrics import Accuracy
from .metrics import Accuracy, DistanceAccuracy


class TaskOCR(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, converter, grad_clip=5.0, hparams={}):
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.converter = converter
        self.grad_clip = grad_clip
        self.hparams = hparams
        self.accuracy = Accuracy(converter=converter)
        self.distance_accuracy =  DistanceAccuracy(converter=converter)
    
    def forward(self, imgs, texts):
        output = self.model(imgs, texts)
        return output
    

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
   
    def shared_step(self, batch, batch_idx):
        images, labels = batch
        labels_encoded, labels_length = self.converter.encode(labels)
        
        preds = self.model(images, labels_encoded[:, :-1]) # align with Attention.forward
        targets = labels_encoded[:, 1:]  # without [GO] Symbol
        
        loss = self.criterion(preds.view(-1, preds.shape[-1]), targets.contiguous().view(-1))
        acc = self.accuracy(preds, labels)
        distance = self.distance_accuracy(preds, labels)
        
        return loss, acc, distance
        
        
    def training_step(self, batch, batch_idx):
        loss, acc, distance = self.shared_step(batch, batch_idx)
        self.log('trn_loss', loss, prog_bar=True, logger=True)
        self.log('trn_acc', acc,  prog_bar=True, logger=True)
        self.log('trn_distance', distance,  prog_bar=True, logger=True)
        
        return loss
        
    def training_epoch_end(self, outs):
        self.log('trn_acc_epoch', self.accuracy.compute(), logger=True)
        self.log('trn_distance_epoch', self.distance_accuracy.compute(), logger=True)
        
    
    def validation_step(self, batch, batch_idx):
        loss, acc, distance = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc,  prog_bar=True, logger=True)
        self.log('val_distance', distance,  prog_bar=True, logger=True)
        
        return loss
    
    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.accuracy.compute(), logger=True)
        self.log('val_distance_epoch', self.distance_accuracy.compute(), logger=True)
    
    def configure_optimizers(self):
        return self.optimizer


class TaskTransformerOCR(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, converter, grad_clip=5.0, hparams={}):
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.converter = converter
        self.grad_clip = grad_clip
        self.hparams = hparams
        self.accuracy = Accuracy(converter=converter)
        self.distance_accuracy =  DistanceAccuracy(converter=converter)
    
    def forward(self, imgs):
        output = self.model(imgs)
        return output
    
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)    
   
    def shared_step(self, batch, batch_idx):
        images, labels = batch
        labels_encoded, labels_length = self.converter.encode(labels)
        
        preds = self.model(images) # align with Attention.forward
        targets = labels_encoded[:, 1:]  # without [GO] Symbol
        
        loss = self.criterion(preds.view(-1, preds.shape[-1]), targets.contiguous().view(-1))
        acc = self.accuracy(preds, labels)
        distance = self.distance_accuracy(preds, labels)
        
        return loss, acc, distance
        
    def training_step(self, batch, batch_idx):
        loss, acc, distance = self.shared_step(batch, batch_idx)
        self.log('trn_loss', loss, prog_bar=True, logger=True)
        self.log('trn_acc', acc,  prog_bar=True, logger=True)
        self.log('trn_distance', distance,  prog_bar=True, logger=True)
        
        return loss
        
    def training_epoch_end(self, outs):
        self.log('trn_acc_epoch', self.accuracy.compute(), logger=True)
        self.log('trn_distance_epoch', self.distance_accuracy.compute(), logger=True)
        
    def validation_step(self, batch, batch_idx):
        loss, acc, distance = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc,  prog_bar=True, logger=True)
        self.log('val_distance', distance,  prog_bar=True, logger=True)
        
        return loss
    
    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.accuracy.compute(), logger=True)
        self.log('val_distance_epoch', self.distance_accuracy.compute(), logger=True)
    
    def configure_optimizers(self):
        return self.optimizer
    
