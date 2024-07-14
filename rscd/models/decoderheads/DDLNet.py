# spatial and temporal feature fusion for change detection of remote sensing images
# STNet11
# Author: xwma
# Time: 2022.11.2

from turtle import forward

import torch 
import torch.nn as nn
import torch.nn.functional as F
import sys
# from models.swintransformer import *
import math

def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes//16, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class SelfAttentionBlock(nn.Module):
    """
    query_feats: (B, C, h, w)
    key_feats: (B, C, h, w)
    value_feats: (B, C, h, w)

    output: (B, C, h, w)
    """
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels,
                 key_query_num_convs, value_out_num_convs):
        super(SelfAttentionBlock, self).__init__()
        self.key_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
        )
        self.query_project = self.buildproject(
            in_channels=query_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs
        )
        self.value_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=value_out_num_convs
        )
        self.out_project = self.buildproject(
            in_channels=transform_channels,
            out_channels=out_channels,
            num_convs=value_out_num_convs
        )
        self.transform_channels = transform_channels

    def forward(self, query_feats, key_feats, value_feats):
        batch_size = query_feats.size(0)

        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous() #(B, h*w, C)

        key = self.key_project(key_feats)
        key = key.reshape(*key.shape[:2], -1) # (B, C, h*w)

        value = self.value_project(value_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous() # (B, h*w, C)

        sim_map = torch.matmul(query, key)
       
        sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1) #(B, h*w, K)
        
        context = torch.matmul(sim_map, value) #(B, h*w, C)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:]) #(B, C, h, w)

        context = self.out_project(context) #(B, C, h, w)
        return context
    def buildproject(self, in_channels, out_channels, num_convs):
        convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_convs-1):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        if len(convs) > 1:
            return nn.Sequential(*convs)
        return convs[0]

class TFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TFF, self).__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xA, xB):
        x_diff = xA - xB

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * xA
        xB = B_weight * xB

        x = self.catconv(torch.cat([xA, xB], dim=1))

        return x

class SFF(nn.Module):
    def __init__(self, in_channel):
        super(SFF, self).__init__()
        self.conv_small = conv_1x1(in_channel, in_channel)
        self.conv_big = conv_1x1(in_channel, in_channel)
        self.catconv = conv_3x3(in_channel*2, in_channel)
        self.attention = SelfAttentionBlock(
            key_in_channels=in_channel,
            query_in_channels = in_channel,
            transform_channels = in_channel // 2,
            out_channels = in_channel,
            key_query_num_convs=2,
            value_out_num_convs=1
        )
    
    def forward(self, x_small, x_big):
        img_size  =x_big.size(2), x_big.size(3)
        x_small = F.interpolate(x_small, img_size, mode="bilinear", align_corners=False)
        x = self.conv_small(x_small) + self.conv_big(x_big)
        new_x = self.attention(x, x, x_big)

        out = self.catconv(torch.cat([new_x, x_big], dim=1))
        return out

class SSFF(nn.Module):
    def __init__(self):
        super(SSFF, self).__init__()
        self.spatial = SpatialAttention()
    def forward(self, x_small, x_big):
        img_shape = x_small.size(2), x_small.size(3)
        big_weight = self.spatial(x_big)
        big_weight = F.interpolate(big_weight, img_shape, mode="bilinear", align_corners=False)
        x_small = big_weight * x_small
        return x_small

class LightDecoder(nn.Module):
    def __init__(self, in_channel, num_class):
        super(LightDecoder, self).__init__()
        self.catconv = conv_3x3(in_channel*4, in_channel)
        self.decoder = nn.Conv2d(in_channel, num_class, 1)
    
    def forward(self, x1, x2, x3, x4):
        x2 = F.interpolate(x2, scale_factor=2, mode="bilinear")
        x3 = F.interpolate(x3, scale_factor=4, mode="bilinear")
        x4 = F.interpolate(x4, scale_factor=8, mode="bilinear")

        out = self.decoder(self.catconv(torch.cat([x1, x2, x3, x4], dim=1)))
        return out



# fca
def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    # MultiSpectralAttentionLayer(planes * 4, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')
    # c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
    # planes * 4 -> channel, c2wh[planes] -> dct_h, c2wh[planes] -> dct_w
    # (64*4,56,56)
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape       # (4,256,64,64)
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:      # dct_h=dct_w=56
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))# (4,256,56,56)
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)        # y:(4,256)

        y = self.fc(y).view(n, c, 1, 1)         # y:(4,256,1,1)
        return x * y.expand_as(x)       # pytorch中的expand_as:扩张张量的尺寸至括号里张量的尺寸 (4,256,64,64)  注意这里是逐元素相乘，不同于qkv的torch.matmul

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    # MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):       # (4,256,56,56)
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight     # weight:(256,56,56)  x:(4,256,56,56)

        result = torch.sum(x, dim=[2,3])        # result:(4,256)
        return result

    def build_filter(self, pos, freq, POS):     # 对应公式中i/j, h/w, H/W   一般是pos即i/j在变
                # self.build_filter(t_x, u_x, tile_size_x)  self.build_filter(t_y, v_y, tile_size_y)
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)        # 为什么是乘以根号2？
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
                # dct_h(height), dct_w(weight), mapper_x, mapper_y, channel(256,512,1024,2048)
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)     # (256,56,56)

        c_part = channel // len(mapper_x)       # c_part = 256/16 = 16

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter


class DDLNet(nn.Module):
    def __init__(self, num_class, channel_list=[64, 128, 256, 512], transform_feat=128):
        super(DDLNet, self).__init__()


        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        self.fca1 = MultiSpectralAttentionLayer(channel_list[0], c2wh[channel_list[0]], c2wh[channel_list[0]],  reduction=16, freq_sel_method = 'top16')
        self.fca2 = MultiSpectralAttentionLayer(channel_list[1], c2wh[channel_list[1]], c2wh[channel_list[1]],  reduction=16, freq_sel_method = 'top16')
        self.fca3 = MultiSpectralAttentionLayer(channel_list[2], c2wh[channel_list[2]], c2wh[channel_list[2]],  reduction=16, freq_sel_method = 'top16')
        self.fca4 = MultiSpectralAttentionLayer(channel_list[3], c2wh[channel_list[3]], c2wh[channel_list[3]],  reduction=16, freq_sel_method = 'top16')

        self.catconv1 = dsconv_3x3(channel_list[0] * 2, out_channel=128)
        self.catconv2 = dsconv_3x3(channel_list[1] * 2, out_channel=128)
        self.catconv3 = dsconv_3x3(channel_list[2] * 2, out_channel=128)
        self.catconv4 = dsconv_3x3(channel_list[3] * 2, out_channel=128)

        self.sff1 = SFF(transform_feat)
        self.sff2 = SFF(transform_feat)
        self.sff3 = SFF(transform_feat)

        self.ssff1 = SSFF()
        self.ssff2 = SSFF()
        self.ssff3 = SSFF()

        self.lightdecoder = LightDecoder(transform_feat, num_class)

        self.catconv = conv_3x3(transform_feat*4, transform_feat)
    
    def forward(self, x):
        featuresA, featuresB = x
        xA1, xA2, xA3, xA4 = featuresA
        xB1, xB2, xB3, xB4 = featuresB

        x1 = self.fca1(xA1)
        x2 = self.fca2(xA2)
        x3 = self.fca3(xA3)
        x4 = self.fca4(xA4)


        x11 = self.fca1(xB1)
        x22 = self.fca2(xB2)
        x33 = self.fca3(xB3)
        x44 = self.fca4(xB4)

        x111 = self.catconv1(torch.cat([x11 - x1, x1], dim=1))
        x222 = self.catconv2(torch.cat([x22 - x2, x2], dim=1))
        x333 = self.catconv3(torch.cat([x33 - x3, x3], dim=1))
        x444 = self.catconv4(torch.cat([x44 - x4, x4], dim=1))

        x1_new = self.ssff1(x444, x111)
        x2_new = self.ssff2(x444, x222)
        x3_new = self.ssff3(x444, x333)

        # print(x1_new.shape)
        # print(x444.shape)
        # print(x111.shape)

        # print(x2_new.shape)
        # print(x444.shape)
        # print(x222.shape)
        x4_new = self.catconv(torch.cat([x444, x1_new, x2_new, x3_new], dim=1))
        # print(x4_new.shape)
        out = self.lightdecoder(x111, x222, x333, x4_new)
        # print(out.shape)
        out = F.interpolate(out, scale_factor=4, mode="bilinear")
        # print(out.shape)
        #return out
        return out


if __name__ == "__main__":
    xa = torch.randn(1, 3, 256, 256)
    xb = torch.randn(1, 3, 256, 256)
    net = DDLNet(2)
    out = net(xa, xb)
    # print(out.shape)
    import thop
    flops, params = thop.profile(net, inputs=(xa,xb,))
    #print(out.shape)
    print(flops/1e9, params/1e6)
