from rscd.models.decoderheads.lgpnet.unet_parts import *

class BCDNET(nn.Module):
    """ Local-Global Pyramid Network (LGPNet) """
    def __init__(self, n_channels, n_classes):
        super(BCDNET, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.conv = TribleConv(128, 64)
        self.final = OutConv(64, n_classes)

    def forward(self, x=[]):
        # out1 = x[0]
        # out2 = x[1]
        feat1 = x[2]
        feat2 = x[3]
        fusionfeats = torch.cat([feat1, feat2], dim=1)

        x = self.conv(fusionfeats)
        logits = self.final(x)
        return logits


class TribleConv(nn.Module):
    """(convolution => [BN] => ReLU) 2次"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.trible_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.trible_conv(x)


if __name__ == '__main__':
    net = BCDNET(n_channels=3, n_classes=1)
    print(net)