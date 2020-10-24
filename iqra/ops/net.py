import torch
import torch.nn as nn

def freeze(net: nn.Module):
    for param in net.parameters():
        param.requires_grad = False

def unfreeze(net: nn.Module):
    for param in net.parameters():
        param.requires_grad = True