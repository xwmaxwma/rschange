
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from rscd.models.backbones.Decom_Backbone import ResNet3D


class AFCD3D_backbone(nn.Module):
    def __init__(self):
        super(AFCD3D_backbone, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet = ResNet3D(resnet)

    def forward(self, imageA, imageB):
        imageA = imageA.unsqueeze(2)
        imageB = imageB.unsqueeze(2)
        x = torch.cat([imageA, imageB], 2)
        size = x.size()[3:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x)
        x = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        return [size, x0, x1, x2, x3, x4]

