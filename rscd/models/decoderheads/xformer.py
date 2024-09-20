import torch
from torch import nn
from torch.cuda.amp import autocast
from rscd.models.decoderheads.vision_lstm import ViLBlock, SequenceTraversal
from torch.nn import functional as F
from functools import partial

class ViLLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.vil = ViLBlock(
            dim= self.dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT
        )
    
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_vil = self.vil(x_flat)
        out = x_vil.transpose(-1, -2).reshape(B, C, *img_dims)

        return out

def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()
        
        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]))

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        
        return x

class XLSTM_axial(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.xlstm_h = ViLLayer(dim = in_channel)
        self.xlstm_w = ViLLayer(dim = in_channel)
        self.xlstm_conv = conv_1x1(in_channel, in_channel)
        self.pos_emb_h = SqueezeAxialPositionalEmbedding(in_channel, 16)
        self.pos_emb_w = SqueezeAxialPositionalEmbedding(in_channel, 16)

    def forward(self, xA, xB):
        x_diff = xA - xB
        B,C,H,W = x_diff.shape
        pos_h = self.pos_emb_h(x_diff.mean(-1))
        pos_w = self.pos_emb_w(x_diff.mean(-2))
        x_xlstm_h = (self.xlstm_h(pos_h) + self.xlstm_h(pos_h.flip([-1])).flip([-1])).reshape(B, C, H, -1)
        x_xlstm_w = (self.xlstm_w(pos_w) + self.xlstm_w(pos_w.flip([-1])).flip([-1])).reshape(B, C, -1, W)
        x_xlstm = self.sigmoid(self.xlstm_conv(x_diff.add(x_xlstm_h.add(x_xlstm_w))))

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * x_xlstm * xA
        xB = B_weight * x_xlstm * xB

        x = self.catconv(torch.cat([xA, xB], dim=1))

        return x

class XLSTM_atten(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.xlstm = ViLLayer(dim = in_channel)

    def forward(self, xA, xB):
        x_diff = xA - xB
        B,C,H,W = x_diff.shape
        x_xlstm = (self.xlstm(x_diff) + self.xlstm(x_diff.flip([-1, -2])).flip([-1, -2]))

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * x_xlstm
        xB = B_weight * x_xlstm

        x = self.catconv(torch.cat([xA, xB], dim=1))

        return x

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
    
class LHBlock(nn.Module):
    def __init__(self, channels_l, channels_h):
        super().__init__()
        self.channels_l = channels_l
        self.channels_h = channels_h
        self.cross_size = 12
        self.cross_kv = nn.Sequential(
            nn.BatchNorm2d(channels_l),
            nn.AdaptiveMaxPool2d(output_size=(self.cross_size, self.cross_size)),
            nn.Conv2d(channels_l, 2 * channels_h, 1, 1, 0)
        )

        self.conv = conv_1x1(channels_l, channels_h)
        self.norm = nn.BatchNorm2d(channels_h)
        
        self.mlp_l = Mlp(in_features=channels_l, out_features=channels_l)
        self.mlp_h = Mlp(in_features=channels_h, out_features=channels_h)

    def _act_sn(self, x):
        _, _, H, W = x.shape
        inner_channel = self.cross_size * self.cross_size
        x = x.reshape([-1, inner_channel, H, W]) * (inner_channel**-0.5)
        x = F.softmax(x, dim=1)
        x = x.reshape([1, -1, H, W])
        return x
    
    def attn_h(self, x_h, cross_k, cross_v):
        B, _, H, W = x_h.shape
        x_h = self.norm(x_h)
        x_h = x_h.reshape([1, -1, H, W])  # n,c_in,h,w -> 1,n*c_in,h,w
        x_h = F.conv2d(x_h, cross_k, bias=None, stride=1, padding=0,
                        groups=B)  # 1,n*c_in,h,w -> 1,n*144,h,w  (group=B)
        x_h = self._act_sn(x_h)
        x_h = F.conv2d(x_h, cross_v, bias=None, stride=1, padding=0,
                        groups=B)  # 1,n*144,h,w -> 1, n*c_in,h,w  (group=B)
        x_h = x_h.reshape([-1, self.channels_h, H,
                        W])  # 1, n*c_in,h,w -> n,c_in,h,w  (c_in = c_out)

        return x_h

    def forward(self, x_l, x_h):
        x_l = x_l + self.mlp_l(x_l)
        x_l_conv = self.conv(x_l)
        x_h = x_h + F.interpolate(x_l_conv, size=x_h.shape[2:], mode='bilinear')

        cross_kv = self.cross_kv(x_l)
        cross_k, cross_v = cross_kv.split(self.channels_h, 1)
        cross_k = cross_k.permute(0, 2, 3, 1).reshape([-1, self.channels_h, 1, 1])  # n*144,channels_h,1,1
        cross_v = cross_v.reshape([-1, self.cross_size * self.cross_size, 1, 1])  # n*channels_h,144,1,1

        x_h = x_h + self.attn_h(x_h, cross_k, cross_v) # [4, 40, 128, 128]
        x_h = x_h + self.mlp_h(x_h)

        return x_h


class CDXLSTM(nn.Module):
    def __init__(self, channels=[40, 80, 192, 384]):
        super().__init__()
        self.channels = channels
        self.fusion0 = XLSTM_axial(channels[0], channels[0])
        self.fusion1 = XLSTM_axial(channels[1], channels[1])
        self.fusion2 = XLSTM_atten(channels[2], channels[2])
        self.fusion3 = XLSTM_atten(channels[3], channels[3])

        self.LHBlock1 = LHBlock(channels[1], channels[0])
        self.LHBlock2 = LHBlock(channels[2], channels[0])
        self.LHBlock3 = LHBlock(channels[3], channels[0])

        self.mlp1 = Mlp(in_features=channels[0], out_features=channels[0])
        self.mlp2 = Mlp(in_features=channels[0], out_features=2)
        self.dwc = dsconv_3x3(channels[0], channels[0])

    def forward(self, inputs):
        featuresA, featuresB = inputs
        # fA_0, fA_1, fA_2, fA_3 = featuresA
        # fB_0, fB_1, fB_2, fB_3 = featuresB
        x_diff_0 = self.fusion0(featuresA[0], featuresB[0]) # [4, 40, 128, 128]
        x_diff_1 = self.fusion1(featuresA[1], featuresB[1]) # [4, 80, 64, 64]
        # x_diff_2 = featuresA[2] -  featuresB[2]
        # x_diff_3 = featuresA[3] -  featuresB[3]
        x_diff_2 = self.fusion2(featuresA[2], featuresB[2]) # [4, 192, 32, 32]
        x_diff_3 = self.fusion3(featuresA[3], featuresB[3]) # [4, 384, 16, 16]
        
        x_h = x_diff_0
        x_h = self.LHBlock1(x_diff_1, x_h) # [4, 40, 128, 128]
        x_h = self.LHBlock2(x_diff_2, x_h)
        x_h = self.LHBlock3(x_diff_3, x_h)

        out = self.mlp2(self.dwc(x_h) + self.mlp1(x_h))

        out = F.interpolate(
            out,
            scale_factor=(4, 4),
            mode="bilinear",
            align_corners=False,
        )
        return out
    
if __name__ == '__main__':
    net = CDXLSTM(channels = [40, 80, 192, 384]).cuda()
    x = [torch.randn(size=(4,40,128,128)).cuda(),
         torch.randn(size=(4,80,64,64)).cuda(),
         torch.randn(size=(4,192,32,32)).cuda(),
         torch.randn(size=(4,384,16,16)).cuda()]
    y = net([x,x])
    print(y.shape)
