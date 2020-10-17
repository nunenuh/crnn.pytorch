import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.metrics import Accuracy


class TaskOCR(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, converter, grad_clip=5.0):
        super().__init__()
        self.model = model
        #self.model = self.model.to(self.device)
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.converter = converter
        self.grad_clip = grad_clip
    
    def forward(self, imgs, texts):
        output = self.model(imgs, texts)
        return output
    

    def backward(self,loss, optimizer, optimizer_idx):
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
   
    def shared_step(self, batch, batch_idx):
        images, texts = batch
        #images = images.to(self.device)

        texts_encoded, texts_length = self.converter.encode(texts)
        #texts_encoded = texts_encoded.to(self.device)
        #texts_length = texts_encoded.to(self.device)
        
        preds = self.model(images, texts_encoded[:, :-1])
        targets = texts_encoded[:, 1:]
        
        loss = self.criterion(preds.view(-1, preds.shape[-1]), targets.contiguous().view(-1))
        
        return loss
        
        
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
#         result = pl.TrainResult(loss)
#         result.log_dict({'trn_loss': loss})
        self.log('trn_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
#         result = pl.EvalResult(checkpoint_on=loss)
#         result.log_dict({'val_loss': loss})
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        return self.optimizer
    
