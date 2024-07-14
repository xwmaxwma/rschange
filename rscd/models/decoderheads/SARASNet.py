import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import models
#from base import BaseModel
#from utils.helpers import initialize_weights
from itertools import chain
#from swin_transformer import SwinTransformer
from einops import rearrange
from torch.hub import load_state_dict_from_url

GlobalAvgPool2D = lambda: nn.AdaptiveAvgPool2d(1)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class Cross_transformer_backbone(nn.Module):
    def __init__(self, in_channels = 48):
        super(Cross_transformer_backbone, self).__init__()
        
        self.to_key = nn.Linear(in_channels * 2, in_channels, bias=False)
        self.to_value = nn.Linear(in_channels * 2, in_channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.gamma_cam_lay3 = nn.Parameter(torch.zeros(1))
        self.cam_layer0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.cam_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.cam_layer2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, input_feature, features):
        Query_features = input_feature
        Query_features = self.cam_layer0(Query_features)       
        key_features = self.cam_layer1(features)
        value_features = self.cam_layer2(features)
        
        QK = torch.einsum("nlhd,nshd->nlsh", Query_features, key_features)
        softmax_temp = 1. / Query_features.size(3)**.5
        A = torch.softmax(softmax_temp * QK, dim=2)
        queried_values = torch.einsum("nlsh,nshd->nlhd", A, value_features).contiguous()
        message = self.mlp(torch.cat([input_feature, queried_values], dim=1))
        
        return input_feature + message

class Cross_transformer(nn.Module):
    def __init__(self, in_channels = 48):
        super(Cross_transformer, self).__init__()
        self.fa = nn.Linear(in_channels , in_channels, bias=False)
        self.fb = nn.Linear(in_channels, in_channels, bias=False)
        self.fc = nn.Linear(in_channels , in_channels, bias=False)
        self.fd = nn.Linear(in_channels, in_channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.gamma_cam_lay3 = nn.Parameter(torch.zeros(1))
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    
    def attention_layer(self, q, k, v, m_batchsize, C, height, width):
        k = k.permute(0, 2, 1)
        energy = torch.bmm(q, k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        out = torch.bmm(attention, v)
        out = out.view(m_batchsize, C, height, width)
        
        return out
        
        
    def forward(self, input_feature, features):    
        fa = input_feature
        fb = features[0]
        fc = features[1]
        fd = features[2]
        

        m_batchsize, C, height, width = fa.size()
        fa = self.fa(fa.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        fb = self.fb(fb.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        fc = self.fc(fc.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        fd = self.fd(fd.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        
        
        qkv_1 = self.attention_layer(fa, fa, fa, m_batchsize, C, height, width)
        qkv_2 = self.attention_layer(fa, fb, fb, m_batchsize, C, height, width)  
        qkv_3 = self.attention_layer(fa, fc, fc, m_batchsize, C, height, width)
        qkv_4 = self.attention_layer(fa, fd, fd, m_batchsize, C, height, width)
        
        atten = self.fuse(torch.cat((qkv_1, qkv_2, qkv_3, qkv_4), dim = 1))
              

        out = self.gamma_cam_lay3 * atten + input_feature

        out = self.to_out(out)
        
        return out


class SceneRelation(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_list,
                 out_channels,
                 scale_aware_proj=True):
        super(SceneRelation, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 1),
                ) for _ in range(len(channel_list))]
            )
        else:
            # 2mlp
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in channel_list:
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )

        self.normalizer = nn.Sigmoid()
        
        

    def forward(self, scene_feature, features: list):
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]

        scene_feats = [op(scene_feature) for op in self.scene_encoder]
        relations = [self.normalizer(sf) * cf for sf, cf in
                         zip(scene_feats, content_feats)]

        
        return relations

class PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]

        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

class Change_detection(nn.Module):
    # Implementing only the object path
    def __init__(self, num_classes=2, use_aux=True, fpn_out=48, freeze_bn=False, **_):
        super(Change_detection, self).__init__()

        f_channels = [64, 128, 256, 512]

        # CNN-backbone     
        self.PPN = PSPModule(f_channels[-1])
        
        # Relation-aware
        self.Cross_transformer_backbone_a3 =  Cross_transformer_backbone(in_channels = f_channels[3])
        self.Cross_transformer_backbone_a2 =  Cross_transformer_backbone(in_channels = f_channels[2])
        self.Cross_transformer_backbone_a1 =  Cross_transformer_backbone(in_channels = f_channels[1])
        self.Cross_transformer_backbone_a0 =  Cross_transformer_backbone(in_channels = f_channels[0])
        self.Cross_transformer_backbone_a33 =  Cross_transformer_backbone(in_channels = f_channels[3])
        self.Cross_transformer_backbone_a22 =  Cross_transformer_backbone(in_channels = f_channels[2])
        self.Cross_transformer_backbone_a11 =  Cross_transformer_backbone(in_channels = f_channels[1])
        self.Cross_transformer_backbone_a00 =  Cross_transformer_backbone(in_channels = f_channels[0])
                
        self.Cross_transformer_backbone_b3 =  Cross_transformer_backbone(in_channels = f_channels[3])
        self.Cross_transformer_backbone_b2 =  Cross_transformer_backbone(in_channels = f_channels[2])
        self.Cross_transformer_backbone_b1 =  Cross_transformer_backbone(in_channels = f_channels[1])
        self.Cross_transformer_backbone_b0 =  Cross_transformer_backbone(in_channels = f_channels[0])
        self.Cross_transformer_backbone_b33 =  Cross_transformer_backbone(in_channels = f_channels[3])
        self.Cross_transformer_backbone_b22 =  Cross_transformer_backbone(in_channels = f_channels[2])
        self.Cross_transformer_backbone_b11 =  Cross_transformer_backbone(in_channels = f_channels[1])
        self.Cross_transformer_backbone_b00 =  Cross_transformer_backbone(in_channels = f_channels[0])


        # Scale-aware
        self.sig = nn.Sigmoid()
        self.gap = GlobalAvgPool2D()
        self.sr1 = SceneRelation(in_channels = f_channels[3], channel_list = f_channels, out_channels = f_channels[3], scale_aware_proj=True)
        self.sr2 = SceneRelation(in_channels = f_channels[2], channel_list = f_channels, out_channels = f_channels[2], scale_aware_proj=True)
        self.sr3 = SceneRelation(in_channels = f_channels[1], channel_list = f_channels, out_channels = f_channels[1], scale_aware_proj=True)
        self.sr4 = SceneRelation(in_channels = f_channels[0], channel_list =f_channels, out_channels = f_channels[0], scale_aware_proj=True)


        # Cross transformer
        self.Cross_transformer1 =  Cross_transformer(in_channels = f_channels[3])
        self.Cross_transformer2 =  Cross_transformer(in_channels = f_channels[2])
        self.Cross_transformer3 =  Cross_transformer(in_channels = f_channels[1])
        self.Cross_transformer4 =  Cross_transformer(in_channels = f_channels[0])


        # Generate change map
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(960 , fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )
        
        self.output_fill = nn.Sequential(
            nn.ConvTranspose2d(fpn_out , fpn_out, kernel_size=2, stride = 2, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_out, num_classes, kernel_size=3, padding=1)
        )
        self.active = nn.Sigmoid() 

    def forward(self, x):
        # CNN-backbone
        features1, features2 = x
        
        features, features11, features22= [], [],[]

        # Relation-aware
        for i in range(len(features1)):
            if i == 0:
                features11.append(self.Cross_transformer_backbone_a00(features1[i] , self.Cross_transformer_backbone_a0(features1[i], features2[i])))
                features22.append(self.Cross_transformer_backbone_b00(features2[i], self.Cross_transformer_backbone_b0(features2[i], features1[i])))
            elif i == 1:
                features11.append(self.Cross_transformer_backbone_a11(features1[i] , self.Cross_transformer_backbone_a1(features1[i], features2[i])))
                features22.append(self.Cross_transformer_backbone_b11(features2[i], self.Cross_transformer_backbone_b1(features2[i], features1[i])))
            elif i == 2:    
                features11.append(self.Cross_transformer_backbone_a22(features1[i] , self.Cross_transformer_backbone_a2(features1[i], features2[i])))
                features22.append(self.Cross_transformer_backbone_b22(features2[i], self.Cross_transformer_backbone_b2(features2[i], features1[i])))
            elif i == 3:    
                features11.append(self.Cross_transformer_backbone_a33(features1[i] , self.Cross_transformer_backbone_a3(features1[i], features2[i])))
                features22.append(self.Cross_transformer_backbone_b33(features2[i], self.Cross_transformer_backbone_b3(features2[i], features1[i])))
          
        # The distance between features from two input images.
        for i in range(len(features1)):
            features.append(abs(features11[i] - features22[i])) 
        features[-1] = self.PPN(features[-1])


        # Scale-aware and cross transformer
        H, W = features[0].size(2), features[0].size(3)
        
        c6 = self.gap(features[-1])   
        c7 = self.gap(features[-2])    
        c8 = self.gap(features[-3])    
        c9 = self.gap(features[-4])   
        
        features1, features2, features3, features4 = [], [], [], []
        features1[:] = [F.interpolate(feature, size=(64, 64), mode='nearest') for feature in features[:]]
        list_3 = self.sr1(c6, features1) 
        fe3 = self.Cross_transformer1(list_3[-1], [list_3[-2], list_3[-3], list_3[-4]]) 
        
        features2[:] = [F.interpolate(feature, size=(64, 64), mode='nearest') for feature in features[:]]
        list_2 = self.sr2(c7, features2) 
        fe2 = self.Cross_transformer2(list_2[-2], [list_2[-1], list_2[-3], list_2[-4]]) 
        
        features3[:] = [F.interpolate(feature, size=(64, 64), mode='nearest') for feature in features[:]]
        list_1 = self.sr3(c8, features3) 
        fe1 = self.Cross_transformer3(list_1[-3], [list_1[-1], list_1[-2], list_1[-4]]) 
        
        features4[:] = [F.interpolate(feature, size=(128, 128), mode='nearest') for feature in features[:]]
        list_0 = self.sr4(c9, features4) 
        fe0 = self.Cross_transformer4(list_0[-4], [list_0[-1], list_0[-2], list_0[-3]]) 

        refined_fpn_feat_list = [fe3, fe2, fe1, fe0]
    
        # Upsampling 
        refined_fpn_feat_list[0] = F.interpolate(refined_fpn_feat_list[0], scale_factor=4, mode='nearest')
        refined_fpn_feat_list[1] = F.interpolate(refined_fpn_feat_list[1], scale_factor=4, mode='nearest')
        refined_fpn_feat_list[2] = F.interpolate(refined_fpn_feat_list[2], scale_factor=4, mode='nearest')
        refined_fpn_feat_list[3] = F.interpolate(refined_fpn_feat_list[3], scale_factor=2, mode='nearest')

        # Generate change map
        x = self.conv_fusion(torch.cat((refined_fpn_feat_list), dim=1))
        x = self.output_fill(x)

        return x


if __name__ == '__main__':
    xa = torch.randn(4, 3, 256, 256)
    xb = torch.randn(4, 3, 256, 256)
    net = Change_detection()
    out = net(xa, xb)
    print(out.shape)