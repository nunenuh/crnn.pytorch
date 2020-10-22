import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.metrics import Accuracy
from .metrics import TextAccuracy


class TaskOCR(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, converter, grad_clip=5.0, hparams={}):
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        #self.model = self.model.to(self.device)
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.converter = converter
        self.grad_clip = grad_clip
        self.hparams = hparams
        self.accuracy = TextAccuracy(converter=converter)
    
    def forward(self, imgs, texts):
        output = self.model(imgs, texts)
        return output
    

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
   
    def shared_step(self, batch, batch_idx):
        images, labels = batch
        #images = images.to(self.device)

        texts_encoded, texts_length = self.converter.encode(labels)
        #texts_encoded = texts_encoded.to(self.device)
        #texts_length = texts_encoded.to(self.device)
        
        preds = self.model(images, texts_encoded[:, :-1]) # align with Attention.forward
        targets = texts_encoded[:, 1:]  # without [GO] Symbol
        
        loss = self.criterion(preds.view(-1, preds.shape[-1]), targets.contiguous().view(-1))
        acc = self.accuracy(preds, labels)
        
        return loss, acc
        
        
    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)
        self.log('trn_loss', loss, prog_bar=True, logger=True)
        self.log('trn_acc_step', acc,  prog_bar=True, logger=True)
        
        
    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy.compute(), logger=True)
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.accuracy.compute(), logger=True)
    
    def configure_optimizers(self):
        return self.optimizer
    



import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.metrics import Accuracy
from .metrics import TextAccuracy


class TaskTransformerOCR(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, converter, grad_clip=5.0, hparams={}):
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        #self.model = self.model.to(self.device)
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.converter = converter
        self.grad_clip = grad_clip
        self.hparams = hparams
        # self.accuracy = TextAccuracy(converter=converter)
    
    def forward(self, imgs, texts):
        output = self.model(imgs, texts)
        return output
    

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
   
    def shared_step(self, batch, batch_idx):
        images, labels = batch
        #images = images.to(self.device)

        texts_encoded, texts_length = self.converter.encode(labels)
        #texts_encoded = texts_encoded.to(self.device)
        #texts_length = texts_encoded.to(self.device)
        
        preds = self.model(images) # align with Attention.forward
        targets = texts_encoded[:, 1:]  # without [GO] Symbol
        
        loss = self.criterion(preds.view(-1, preds.shape[-1]), targets.contiguous().view(-1))
        # acc = self.accuracy(preds, labels)
        
        return loss
        
        
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('trn_loss', loss, prog_bar=True, logger=True)
        # self.log('trn_acc_step', acc,  prog_bar=True, logger=True)
        
        
    # def training_epoch_end(self, outs):
    #     self.log('train_acc_epoch', self.accuracy.compute(), logger=True)
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    # def validation_epoch_end(self, outs):
    #     self.log('val_acc_epoch', self.accuracy.compute(), logger=True)
    
    def configure_optimizers(self):
        return self.optimizer
    
