import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.metrics import Metric
from . import functional as MF


class Accuracy(Metric):
    def __init__(self, converter, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.converter = converter

    def update(self, preds: torch.Tensor, labels, max_length=25):
        batch_size = preds.size(0)
        preds_str = MF.prediction_string(preds, self.converter, max_length=max_length)
        correct = MF.count_text_correct(preds_str, labels)
        
        self.correct += correct
        self.total += batch_size
        
    def compute(self):
        return self.correct.float() / self.total 


class DistanceAccuracy(Metric):
    def __init__(self, converter, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("norm_ed", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.converter = converter

    def update(self, preds: torch.Tensor, labels, max_length=25):
        batch_size = preds.size(0)
        preds_str = MF.prediction_string(preds, self.converter, max_length=max_length)
        norm_ed = MF.count_norm_distance(preds_str, labels)
        
        self.norm_ed += norm_ed
        self.total += batch_size
        
    def compute(self):
        return self.norm_ed.float() / self.total 
 
 
    

if __name__ == "__main__":
    pass