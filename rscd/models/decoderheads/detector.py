import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )

class changedetector(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.mlp1 = Mlp(in_features = in_channel, out_features = in_channel)
        self.mlp2 = Mlp(in_features = in_channel, out_features=2)
        self.dwc = dsconv_3x3(in_channel, in_channel)

    def forward(self, x):
        x1 = self.mlp1(x)
        x_d = self.dwc(x1)
        x_out = self.mlp2(x1 + x_d)
        x_out = F.interpolate(
            x_out,
            scale_factor=(4,4),
            mode="bilinear",
            align_corners=False,
        )
        return x_out
