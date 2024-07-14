import torch
import torch.nn as nn
from rscd.models.backbones.lgpnet.BFE_DPN import BFExtractor

class LGPNet_a(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BFExtractor(n_channels=3, n_classes=2)
    def forward(self, xA, xB):
        list = []  # 0: out1,1: out2,2: feat1,3: feat2
        out1, feat1 = self.backbone(xA)
        out2, feat2 = self.backbone(xB)
        list.append(out1)
        list.append(out2)
        list.append(feat1)
        list.append(feat2)
        return list
