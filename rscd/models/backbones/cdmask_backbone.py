import torch 
import torch.nn as nn
from rscd.models.backbones.seaformer import *

class Seaformer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.backbone = SeaFormer_L(pretrained=True)
        self.channels = channels

    def forward(self, xA, xB):
        featuresA = self.backbone(xA)
        featuresB = self.backbone(xB)

        return [featuresA, featuresB]