import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.metrics import Metric
import re

class TextAccuracy(Metric):
    def __init__(self, converter, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.add_state("confidence")
        self.converter = converter

    def update(self, preds: torch.Tensor, labels, max_length=25):
        batch_size = preds.size(0)
        preds_str, preds_max_prob = self.preds_preprocess(preds, max_length=max_length)
        correct, confidence  = self.calculate_correct(preds_str, preds_max_prob, labels)
        
        # self.confidence.append(confidence)
        self.correct += correct
        self.total += batch_size
        
    def preds_preprocess(self, preds, max_length):
        batch_size = preds.size(0)
        length = torch.IntTensor([max_length] * batch_size)
        
        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, length)
        
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        
        return preds_str, preds_max_prob 
        
    def calculate_correct(self, preds_str, preds_max_prob, labels):
        n_correct = 0
        confidence_scores = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            pred, gt, pred_eos, gt_eos = self.prune_eos(pred, gt)
            pred, gt = self.case_sensitive_eval(pred, gt)
            if pred == gt: n_correct += 1
            
            try:
                pred_max_prob = pred_max_prob[:pred_eos]
                confidence = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_scores.append(confidence)
            
        return n_correct, confidence_scores
        
    
    def prune_eos(self, pred, gt):
        # prune after "end of sentence" token ([s])
        pred_eos, gt_eos = pred.find('[s]'), gt.find('[s]')
        pred, gt  = pred[:pred_eos], gt[:gt_eos]
        return pred, gt, pred_eos, gt_eos
        
        
    def case_sensitive_eval(self, pred, gt):
        pred, gt = pred.lower(), gt.lower()
        alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
        out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
        pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
        gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)
        return pred, gt
        

    def compute(self):
        return self.correct.float() / self.total * 100
    
    
    

if __name__ == "__main__":
    pass