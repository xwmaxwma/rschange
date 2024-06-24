import torch
from torch import nn
from addict import Dict

from rscd.models.decoderheads.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder4ScalesFASeg
from rscd.models.decoderheads.transformer_decoder import MultiScaleMaskedTransformerDecoder_OurDH_v4,MultiScaleMaskedTransformerDecoder_OurDH_v5

from torch.nn import functional as F

class MaskFormerHead(nn.Module):
    def __init__(self, input_shape,
                 num_classes = 1,
                 num_queries = 10,
                 dec_layers = 10
                 ):        
        super().__init__()        
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.dec_layers = dec_layers
        self.pixel_decoder = self.pixel_decoder_init(input_shape)
        self.predictor = self.predictor_init()
        
    def pixel_decoder_init(self, input_shape):
        common_stride = 4 # cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        transformer_dropout = 0 # cfg.MODEL.MASK_FORMER.DROPOUT
        transformer_nheads = 8 # cfg.MODEL.MASK_FORMER.NHEADS
        transformer_dim_feedforward = 1024
        transformer_enc_layers = 4 # cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        conv_dim = 256 # cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = 256 # cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        transformer_in_features = ["res3", "res4", "res5"] # cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES # ["res3", "res4", "res5"]

        pixel_decoder = MSDeformAttnPixelDecoder4ScalesFASeg(input_shape,
                                                transformer_dropout,
                                                transformer_nheads,
                                                transformer_dim_feedforward,
                                                transformer_enc_layers,
                                                conv_dim,
                                                mask_dim,
                                                transformer_in_features,
                                                common_stride)
        return pixel_decoder

    def predictor_init(self):
        in_channels = 256 # cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        num_classes = self.num_classes # cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        hidden_dim = 256 # cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        num_queries = self.num_queries # cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        nheads = 8 # cfg.MODEL.MASK_FORMER.NHEADS
        dim_feedforward = 1024 # cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        dec_layers = self.dec_layers - 1 # cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        pre_norm = False # cfg.MODEL.MASK_FORMER.PRE_NORM
        mask_dim = 256 # cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        enforce_input_project = False
        mask_classification = True
        predictor = MultiScaleMaskedTransformerDecoder_OurDH_v5(in_channels, 
                                                        num_classes, 
                                                        mask_classification,
                                                        hidden_dim,
                                                        num_queries,
                                                        nheads,
                                                        dim_feedforward,
                                                        dec_layers,
                                                        pre_norm,
                                                        mask_dim,
                                                        enforce_input_project)
        return predictor

    def forward(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features, pos_list_2d = self.pixel_decoder.forward_features(features)   
        predictions = self.predictor(multi_scale_features, mask_features, mask, pos_list_2d)  
        return predictions

def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

class SaELayer(nn.Module):
    def __init__(self, in_channel, reduction=32):
        super(SaELayer, self).__init__()
        assert in_channel>=reduction and in_channel%reduction==0,'invalid in_channel in SaElayer'
        self.reduction = reduction
        self.cardinality=4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #cardinality 1
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel,in_channel//self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 2
        self.fc2 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 3
        self.fc3 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 4
        self.fc4 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_channel//self.reduction*self.cardinality, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1 = self.fc1(y)
        y2 = self.fc2(y)
        y3 = self.fc3(y)
        y4 = self.fc4(y)
        y_concate = torch.cat([y1,y2,y3,y4],dim=1)
        y_ex_dim = self.fc(y_concate).view(b,c,1,1)

        return y_ex_dim.expand_as(x)

class TFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TFF, self).__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.senetv2 = SaELayer(in_channel)

    def forward(self, xA, xB):
        x_diff = xA - xB
        x_weight = self.senetv2(x_diff)

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * xA * x_weight
        xB = B_weight * xB * x_weight

        x = self.catconv(torch.cat([xA, xB], dim=1))

        return x
    
class MaskFormerModel_sea_ourDH(nn.Module):
    def __init__(self, channels,
                 num_classes = 1,
                 num_queries = 10,
                 dec_layers = 14):
        super().__init__()
        self.channels = channels
        self.backbone_feature_shape = dict()
        for i, channel in enumerate(self.channels):
            self.backbone_feature_shape[f'res{i+2}'] = Dict({'channel': channel, 'stride': 2**(i+2)})

        self.tff1 = TFF(self.channels[0], self.channels[0])
        self.tff2 = TFF(self.channels[1], self.channels[1])
        self.tff3 = TFF(self.channels[2], self.channels[2])
        self.tff4 = TFF(self.channels[3], self.channels[3])

        self.sem_seg_head = MaskFormerHead(self.backbone_feature_shape, num_classes, num_queries, dec_layers)

    def semantic_inference(self, mask_cls, mask_pred):
        # mask_cls = F.softmax(mask_cls, dim=-1)
        mask_cls = F.softmax(mask_cls, dim=-1)[...,1:]
        mask_pred = mask_pred.sigmoid()  
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred).detach()
        b, c, h, w = semseg.shape
        for i in range(b):
            for j in range(c):
                minval = semseg[i, j].min()
                maxval = semseg[i, j].max()
                semseg[i, j] = (semseg[i, j] - minval) / (maxval - minval)
        return semseg

    def forward(self, inputs):
        featuresA, featuresB =inputs
        features = [self.tff1(featuresA[0], featuresB[0]),
                self.tff2(featuresA[1], featuresB[1]),
                self.tff3(featuresA[2], featuresB[2]),
                self.tff4(featuresA[3], featuresB[3]),]
        features = {
            'res2': features[0],
            'res3': features[1],
            'res4': features[2],
            'res5': features[3]
        }

        outputs = self.sem_seg_head(features)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        mask_pred_results = F.interpolate(
            mask_pred_results,
            scale_factor=(4,4),
            mode="bilinear",
            align_corners=False,
        )
        pred_masks = self.semantic_inference(mask_cls_results, mask_pred_results)

        return [pred_masks, outputs]
    
