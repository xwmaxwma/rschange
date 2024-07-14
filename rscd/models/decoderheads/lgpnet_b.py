import torch
import torch.nn as nn
from rscd.models.decoderheads.lgpnet.BCDNET import BCDNET   

class LGPNet_b(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = BCDNET(n_channels=3, n_classes=2)

    def forward(self, x):
        pred = self.net(x)
        return pred