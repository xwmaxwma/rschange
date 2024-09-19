import torch
from rscd.models.backbones import Decompose


class ResNet3D(torch.nn.Module):
    def __init__(self, resnet2d):
        super(ResNet3D, self).__init__()
        self.conv1 = Decompose.Decompose_conv(resnet2d.conv1, time_dim=3, time_padding=1, center=True)
        self.bn1 = Decompose.Decompose_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = Decompose.Decompose_pool(resnet2d.maxpool, time_dim=1, time_padding=0, time_stride=1)

        self.layer1 = Decompose_layer(resnet2d.layer1)
        self.layer2 = Decompose_layer(resnet2d.layer2)
        self.layer3 = Decompose_layer(resnet2d.layer3)
        self.layer4 = Decompose_layer(resnet2d.layer4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x




def Decompose_layer(reslayer2d):
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = Bottleneck3d(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class Bottleneck3d(torch.nn.Module):
    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        self.conv1 = Decompose.Decompose_conv(bottleneck2d.conv1, time_dim=3, time_padding=1,
                                                time_stride=1, center=True)
        self.bn1 = Decompose.Decompose_norm(bottleneck2d.bn1)

        self.conv2 = Decompose.Decompose_conv(bottleneck2d.conv2, time_dim=3, time_padding=1,
                                                time_stride=1, center=True)
        self.bn2 = Decompose.Decompose_norm(bottleneck2d.bn2)

        # self.conv3 = Decompose.Decompose_conv(bottleneck2d.conv3, time_dim=1, center=True)
        # self.bn3 = Decompose.Decompose_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = Decompose_downsample(bottleneck2d.downsample, time_stride=1)
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        out = out + residual
        out = self.relu(out)
        return out


def Decompose_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(
        Decompose.inflate_conv(downsample2d[0], time_dim=1, time_stride=time_stride, center=True),
        Decompose.Decompose_norm(downsample2d[1]))
    return downsample3d





