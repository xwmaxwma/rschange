import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet18stem': 'https://download.openmmlab.com/pretrain/third_party/resnet18_v1c-b5776b93.pth',
    'resnet50stem': 'https://download.openmmlab.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth',
    'resnet101stem': 'https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth',
}

def conv3x3(in_planes, outplanes, stride=1):
    # 带padding的3*3卷积
    return nn.Conv2d(in_planes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    """
    Basic Block for Resnet
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=dilation, 
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = out + residual
        out = self.relu_inplace(out)

        return out

class Resnet(nn.Module):
    def __init__(self, block, layers, out_stride=8, use_stem=False, stem_channels=64, in_channels=3):
        self.inplanes = 64
        super(Resnet, self).__init__()
        outstride_to_strides_and_dilations = {
            8: ((1, 2, 1, 1), (1, 1, 2, 4)),
            16: ((1, 2, 2, 1), (1, 1, 1, 2)),
            32: ((1, 2, 2, 2), (1, 1, 1, 1)),
        }
        stride_list, dilation_list = outstride_to_strides_and_dilations[out_stride]

        self.use_stem = use_stem
        if use_stem:
            self.stem = nn.Sequential(
                conv3x3(in_channels, stem_channels//2, stride=2),
                nn.BatchNorm2d(stem_channels//2),
                nn.ReLU(inplace=False),

                conv3x3(stem_channels//2, stem_channels//2),
                nn.BatchNorm2d(stem_channels//2),
                nn.ReLU(inplace=False),

                conv3x3(stem_channels//2, stem_channels),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=False)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(stem_channels)
            self.relu = nn.ReLU(inplace=False)
        
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks=layers[0], stride=stride_list[0], dilation=dilation_list[0])
        self.layer2 = self._make_layer(block, 128, blocks=layers[1], stride=stride_list[1], dilation=dilation_list[1])
        self.layer3 = self._make_layer(block, 256, blocks=layers[2], stride=stride_list[2], dilation=dilation_list[2])
        self.layer4 = self._make_layer(block, 512, blocks=layers[3], stride=stride_list[3], dilation=dilation_list[3])

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, contract_dilation=True):
        downsample = None
        dilations = [dilation] * blocks

        if contract_dilation and dilation > 1: dilations[0] = dilation // 2
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilations[0], downsample=downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilations[i]))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_stem:
            x = self.stem(x)
        else:
            x = self.relu(self.bn1(self.conv1(x)))

        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        outs = [x1, x2, x3, x4]

        return tuple(outs)

def get_resnet18(pretrained=True):
    model = Resnet(BasicBlock, [2, 2, 2, 2], out_stride=32)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet18'])
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    return model

def get_resnet50_OS8(pretrained=True):
    model = Resnet(Bottleneck, [3, 4, 6, 3], out_stride=8, use_stem=True)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet50stem'])
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    return model

def get_resnet50_OS32(pretrained=True):
    model = Resnet(Bottleneck, [3, 4, 6, 3], out_stride=32, use_stem=False)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet50'])
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    return model

if __name__ == "__main__":
    model = get_resnet50_OS32()
    x = torch.randn(4, 3, 256, 256)
    x = model(x)[-1]
    print(x.shape)