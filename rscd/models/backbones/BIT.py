import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

from rscd.models.backbones import resnet_bit

class BIT_Backbone(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(BIT_Backbone, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = resnet_bit.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet = resnet_bit.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = resnet_bit.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError

        self.upsamplex2 = nn.Upsample(scale_factor=2)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, output_nc, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        return [x1, x2]

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x

def BIT_backbone_func(cfg):
    net = BIT_Backbone(input_nc=cfg.input_nc, 
                       output_nc=cfg.output_nc,
                       resnet_stages_num=cfg.resnet_stages_num, 
                       backbone=cfg.backbone,
                       if_upsample_2x=cfg.if_upsample_2x)
    return net

if __name__ == '__main__':
    x1 = torch.rand(4, 3, 512, 512)
    x2 = torch.rand(4, 3, 512, 512)
    cfg = dict(
        type = 'BIT_Backbone',
        input_nc=3, 
        output_nc=32, 
        resnet_stages_num=4,
        backbone='resnet18',
        if_upsample_2x=True,
    )
    from munch import DefaultMunch 
    cfg = DefaultMunch.fromDict(cfg)
    model = BIT_backbone_func(cfg)
    model.eval()
    print(model)
    outs = model(x1, x2)
    print('BIT', outs)
    for out in outs:
        print(out.shape)