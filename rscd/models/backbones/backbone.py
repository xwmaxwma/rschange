import torch 
import torch.nn as nn
from rscd.models.backbones.seaformer import *
from rscd.models.backbones.resnet import get_resnet18, get_resnet50_OS32, get_resnet50_OS8
from rscd.models.backbones.swintransformer import *

class Base(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name == 'Seaformer':
            self.backbone = SeaFormer_L(pretrained=True)
        elif name == 'Resnet18':
            self.backbone = get_resnet18(pretrained=True)
        elif name == 'Swin':
            self.backbone = swin_tiny(True)

    def forward(self, xA, xB):
        featuresA = self.backbone(xA)
        featuresB = self.backbone(xB)

        return [featuresA, featuresB]
