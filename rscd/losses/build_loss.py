import torch
import torch.nn as nn
from rscd.losses.loss_func import CELoss, FocalLoss, dice_loss, BCEDICE_loss, LOVASZ
from rscd.losses.mask2formerLoss import Mask2formerLoss
from rscd.losses.RSMambaLoss import FCCDN_loss_without_seg

class myLoss(nn.Module):
    def __init__(self, param, loss_name=['CELoss'], loss_weight=[1.0], **kwargs):
        super(myLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss = list()
        for _loss in loss_name:
            self.loss.append(eval(_loss)(**param[_loss],**kwargs))
    
    def forward(self, preds, target):
        loss = 0
        for i in range(0, len(self.loss)):
            loss += self.loss[i](preds, target) * self.loss_weight[i]
        return loss
    
def build_loss(cfg):
    loss_type = cfg.pop('type')
    obj_cls = eval(loss_type)
    obj = obj_cls(**cfg)
    return obj
