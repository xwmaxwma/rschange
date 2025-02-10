import torch 
import torch.nn as nn
import torch.nn.functional as F
from rscd.models.backbones.seaformer_vmanba import SeaFormer_L

from rscd.models.backbones.cdloma import SS2D

class cdlamba(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.backbone = SeaFormer_L(pretrained=True)
        self.channels = channels
        
        self.css = nn.ModuleList()

        for i in range(20):
            self.css.append(SS2D(d_model = self.channels[i // 5], channel_first=True, stage_num= i // 5, depth_num= i % 5).cuda())
            
        input_proj_list = []

        for i in range(4):
            in_channels = self.channels[i]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1),
                nn.GroupNorm(32, in_channels * 2),
            ))
            
        self.input_proj = nn.ModuleList(input_proj_list)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, xA, xB):
        inA, inB = xA, xB
        css_out = []
        for i in range(4):
            fA = self.backbone(inA, i)
            fB = self.backbone(inB, i)
 
            f = torch.concat([fA, fB], 1)

            f1 = self.css[i * 5](f)
            f2 = self.css[i * 5 + 1](f)
            f3 = self.css[i * 5 + 2](f)
            f4 = self.css[i * 5 + 3](f)
            f5 = self.css[i * 5 + 4](f)

            f = self.input_proj[i](f1 + f2 + f3 + f4 + f5)

            cdaA, cdaB = torch.split(f, self.channels[i], 1)
            css_out.append(cdaA - cdaB)
            inA, inB = fA + cdaA, fB + cdaB

        for i in range(1, 4):
            css_out[i] = F.interpolate(
                css_out[i],
                scale_factor=(2 ** i, 2 ** i),
                mode="bilinear",
                align_corners=False,
            )

        extract_out = torch.concat(css_out, dim=1)
        
        return extract_out
