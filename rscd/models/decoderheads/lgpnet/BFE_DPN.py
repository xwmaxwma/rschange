from rscd.models.decoderheads.lgpnet.unet_parts import *
from rscd.models.decoderheads.lgpnet.ChannelAttention import ChannelAttention
from rscd.models.decoderheads.lgpnet.SpatialPyramidModule import SPM
from rscd.models.decoderheads.lgpnet.FeaturePyramidModule import FPM
from rscd.models.decoderheads.lgpnet.PositionAttentionModule import PAM

class BFExtractor(nn.Module):
    """ Full assembly of the parts to form the complete network """
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(BFExtractor, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.pam = PAM(1024)

        self.psp = SPM(1024, 1024, sizes=(1, 2, 3, 6))
        self.fpa = FPM(1024)
        self.drop = nn.Dropout2d(p=0.2)
        self.ca = ChannelAttention(in_channels=1024)
        self.conv1x1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)

        self.up1 = Up(1024, 512, bilinear)
        self.ca1 = ChannelAttention(in_channels=512)
        self.up2 = Up(512, 256, bilinear)
        self.ca2 = ChannelAttention(in_channels=256)
        self.up3 = Up(256, 128, bilinear)
        self.ca3 = ChannelAttention(in_channels=128)
        self.up4 = Up(128, 64, bilinear)
        self.ca4 = ChannelAttention(in_channels=64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        pam_x5 = self.pam(x5)

        # Spatial Pyramid Module
        psp = self.psp(pam_x5)
        pspdrop = self.drop(psp)
        capsp = self.ca(pspdrop)
        capsp = self.conv1x1(capsp)

        # Feature Pyramid Attention Module
        fpa = self.fpa(pam_x5)
        fpadrop = self.drop(fpa)
        cafpa = self.ca(fpadrop)
        cafpa = self.conv1x1(cafpa)

        ca_psp_fpa = torch.cat([capsp, cafpa], dim=1)

        x = self.up1(ca_psp_fpa, x4)
        x = self.ca1(x)
        x = self.up2(x, x3)
        x = self.ca2(x)
        x = self.up3(x, x2)
        x = self.ca3(x)
        x = self.up4(x, x1)
        feats = self.ca4(x)
        logits = self.outc(x)
        return logits, feats

if __name__ == '__main__':
    net = BFExtractor(n_channels=3, n_classes=1)
    print(net)