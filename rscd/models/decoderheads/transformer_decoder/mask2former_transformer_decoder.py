# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .position_encoding import PositionEmbeddingSine
from .DEACA import DEACA_attention, DEACA_attention_v3, rcda_rebuild

import math
import matplotlib.pyplot as plt
import os
from PIL import Image

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        mask_classification=True,  
        hidden_dim=256,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=10,
        pre_norm=False,
        mask_dim=256,
        enforce_input_project=False
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask[:,:] = False
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


class MultiScaleMaskedTransformerDecoder_mp(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        mask_classification=True,  
        hidden_dim=256,
        num_queries=100,
        gt_queries = 100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=10,
        pre_norm=False,
        mask_dim=256,
        enforce_input_project=False
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.num_classes = num_classes

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        self.gt_queries = gt_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.gt_query_embed = nn.Embedding(gt_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim)

    def prepare_for_dn_v5(self, mask_features, size_list, targets, bs):
        l = [t['labels'] for t in targets]
        num = [len(b) for b in l]
        if max(num) == 0:
            return None

        scalar, noise_scale = 1, 0.2
        single_pad = max_num = self.gt_queries

        head_dn = False
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size

        padding = torch.zeros([bs, pad_size, self.query_feat.weight.shape[-1]]).to(device=mask_features.device)
        padding_mask = torch.ones([bs, pad_size, size_list[0][0]*size_list[0][1]]).to(device=mask_features.device).bool()
        masks = torch.cat([F.interpolate(targets[i]['masks'].float().unsqueeze(1), size=size_list[0], mode="area").flatten(1)<=1e-8
                 for i in range(len(targets)) if len(targets[i]['masks'])>0]).repeat(scalar, 1)
        if head_dn:
            masks = masks[:, None].repeat(1, 8, 1)
            areas = (~masks).sum(2)
            noise_ratio = areas * noise_scale / (size_list[0][0] * size_list[0][1])
            delta_mask = torch.rand_like(masks, dtype=torch.float) < noise_ratio[:,:, None]
            masks = torch.logical_xor(masks, delta_mask) #[q,h,h*w]
        else:
            areas= (~masks).sum(1)
            noise_ratio=areas*noise_scale/(size_list[0][0]*size_list[0][1])
            delta_mask=torch.rand_like(masks,dtype=torch.float)<noise_ratio[:,None]
            masks=torch.logical_xor(masks,delta_mask)


        # torch.sum(dim=)

        labels=torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(scalar, 1).view(-1)
        # known_labels = self.label_enc(known_labels)

        # dn_label_noise_ratio = 0.2
        knwon_labels_expand = known_labels.clone()
        # if dn_label_noise_ratio > 0:
        #     prob = torch.rand_like(knwon_labels_expand.float())
        #     chosen_indice = prob < dn_label_noise_ratio
        #     new_label = torch.randint_like(knwon_labels_expand[chosen_indice], 0,
        #                                    self.num_classes)  # randomly put a new one here
        #     # gt_labels_expand.scatter_(0, chosen_indice, new_label)
        #     knwon_labels_expand[chosen_indice] = new_label

        known_labels = self.label_enc(knwon_labels_expand)
        # noised_known_features = known_labels
        noised_known_features = known_labels


        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')
        # import pdb;pdb.set_trace()
        map_known_indices = torch.cat([torch.tensor(range(num)) for num in [len(t['labels']) for t in targets]])  # [1,2, 1,2,3]
        # print(map_known_indices)
        # print(boxes)
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(scalar)]).long().to(device=mask_features.device)

        # for i in range(bs):
        padding[(known_bid, map_known_indices)] = noised_known_features
        if head_dn:
            padding_mask=padding_mask.unsqueeze(2).repeat([1,1,8,1]) #[bs,q,h,h*w]
            padding_mask[(known_bid, map_known_indices)] = masks
            padding_mask=padding_mask.transpose(1,2)
        else:
            padding_mask[(known_bid, map_known_indices)]=masks
            padding_mask=padding_mask.unsqueeze(1).repeat([1,8,1,1])

        res = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        res = torch.cat([padding.transpose(0, 1), res], dim=0)
        ###################

        outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,attn_mask_target_size=size_list[0])
        attn_mask_=attn_mask_.view([bs,8,-1,attn_mask_.shape[-1]])
        attn_mask_[:,:,:-self.num_queries]=padding_mask
        attn_mask_=attn_mask_.flatten(0,1)
        #####################

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to(device=mask_features.device) < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return (known_bid,map_known_indices), res, attn_mask, dn_args_,outputs_class, outputs_mask, attn_mask_

    def gen_mask_dn(self,size_list,known_bid,map_known_indices, targets, bs, mask_features):
        l = [t['labels'] for t in targets]
        num = [len(b) for b in l]
        if max(num) == 0:
            return None
        
        scalar, noise_scale = 1, 0.2
        max_num = self.gt_queries

        head_dn = False
        pad_size = scalar * max_num
        dn_args_ = dict()
        dn_args_['max_num'] = max_num
        dn_args_['pad_size'] = pad_size

        padding_mask = torch.ones([bs, pad_size, size_list[0]*size_list[1]]).to(device=mask_features.device).bool()
        masks = torch.cat([F.interpolate(targets[i]['masks'].float().unsqueeze(1), size=size_list[0], mode="area").flatten(1)<=1e-8
                 for i in range(len(targets)) if len(targets[i]['masks'])>0]).repeat(scalar, 1)
        if head_dn:
            masks = masks[:, None].repeat(1, 8, 1)
            areas = (~masks).sum(2)
            noise_ratio = areas * noise_scale / (size_list[0] * size_list[1])
            delta_mask = torch.rand_like(masks, dtype=torch.float) < noise_ratio[:,:, None]
            masks = torch.logical_xor(masks, delta_mask) #[q,h,h*w]
        else:
            areas= (~masks).sum(1)
            noise_ratio=areas*noise_scale/(size_list[0]*size_list[1])
            delta_mask=torch.rand_like(masks,dtype=torch.float)<noise_ratio[:,None]
            masks=torch.logical_xor(masks,delta_mask)

        if head_dn:
            padding_mask=padding_mask.unsqueeze(2).repeat([1,1,8,1]) #[bs,q,h,h*w]
            padding_mask[(known_bid, map_known_indices)] = masks
            padding_mask=padding_mask.transpose(1,2)
        else:
            padding_mask[(known_bid, map_known_indices)]=masks
            padding_mask=padding_mask.unsqueeze(1).repeat([1,8,1,1])

        return padding_mask
    
    def postprocess_for_dn(self, predictions_class, predictions_mask):
        n_lys = len(predictions_class)
        dn_predictions_class, predictions_class = [predictions_class[i][:, :-self.num_queries] for i in range(n_lys)], \
                                                  [predictions_class[i][:, -self.num_queries:] for i in range(n_lys)]
        dn_predictions_mask, predictions_mask = [predictions_mask[i][:, :-self.num_queries] for i in range(n_lys)], \
                                                [predictions_mask[i][:, -self.num_queries:] for i in range(n_lys)]
        return predictions_class, predictions_mask, dn_predictions_class, dn_predictions_mask
    
    def forward(self, x, mask_features, targets, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        if not targets == None:
            res= self.prepare_for_dn_v5(mask_features, size_list, targets, bs)
            if res == None:
                query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
                tgt_mask = None
                output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
                outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                    attn_mask_target_size=size_list[0])
            else:
                query_embed, output, tgt_mask, dn_args_,outputs_class, outputs_mask, attn_mask=res
                known_bid,map_known_indices=query_embed
                query_embed=torch.cat([self.gt_query_embed.weight.unsqueeze(1).repeat(1, bs, 1), 
                                       self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)], dim=0) 
        else:
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            tgt_mask = None
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                attn_mask_target_size=size_list[0])
        # QxNxC
        # query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        dn_predictions_class = []
        dn_predictions_mask = []

        # prediction heads on learnable query features
        # outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])

            if (not targets == None) and (not tgt_mask == None):
                padding_mask=self.gen_mask_dn(size_list=size_list[(i + 1) % self.num_feature_levels],known_bid=known_bid,map_known_indices=map_known_indices, targets=targets, bs=bs, mask_features=mask_features)
                attn_mask = attn_mask.view([bs, 8, -1, attn_mask.shape[-1]])
                attn_mask[:, :, :-self.num_queries] = padding_mask
                attn_mask = attn_mask.flatten(0, 1)
        
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        if not (tgt_mask is None):
            predictions_class, predictions_mask, dn_predictions_class, dn_predictions_mask = \
                self.postprocess_for_dn(predictions_class, predictions_mask)

            dn_out = {
                'pred_logits': dn_predictions_class[-1],
                'pred_masks': dn_predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(
                    dn_predictions_class if self.mask_classification else None, dn_predictions_mask
                ),
                'dn_args': dn_args_

            }
        else:
            dn_out = None
            predictions_class[-1]+=self.label_enc.weight[0,0]*0.0

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            'dn_out': dn_out
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask[:,:] = False
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

class CrossAttentionLayerFASeg(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        # We return the generated attention map
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """

    B, C, H, W = x.shape

    try:
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    except:
        print('hey')
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size, window_size, C)

    return windows

class MultiScaleMaskedTransformerDecoderFASeg(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            mask_classification=True,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=10,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
            omega_rate=32
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        # self.pos_encoding = nn.Embedding(32 * 32, hidden_dim)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.hidden_dim = hidden_dim

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayerFASeg(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding, we add one extra high-resolution scale
        self.num_feature_levels = 4
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # projection layer for generate positional queries
        self.adapt_pos2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # The rate for selecting high-resolution areas
        self.omega_rate = omega_rate

    def forward(self, x, mask_features, mask=None, pos_list_2d=None):
        # x is a list of multi-scale feature
        # assert len(x) == self.num_feature_levels
        src = []
        pos = []
        pos_2d = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])

            # Use the same positional encoding in the pixel decoder
            pos_encoding = pos_list_2d[i]
            pos_2d.append(pos_encoding)

            pos.append(pos_encoding.flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        fg_src = self.input_proj[-1](x[-1]) + self.level_embed.weight[-1][None, :, None, None]
        fg_pos = pos_2d[-1]

        n, bs, c = src[0].shape

        # QxNxC
        query_embed_params = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, pooling_mask = self.forward_prediction_heads(output, mask_features,
                                                                               attn_mask_target_size=size_list[0])

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        # We generate dynamic positional queries for the first layer based on the predicted
        # positions of the first prediction head in Mask2former
        query_embed = (pooling_mask.permute(1, 2, 0).unsqueeze(-1) * pos[0].unsqueeze(0)).sum(1) / \
                     (pooling_mask.sum(-1).permute(1,0)[..., None] + 1e-6)

        query_embed = self.adapt_pos2d(query_embed + query_embed_params)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first

            if i % 4 == 0:

                # Keep the attention in the first cross-attention layer
                output, attn = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed,
                )

                cg_attn = attn

            elif i % 4 == 3:

                # Select top-k areas in high-resolution attention maps
                _, indexes = cg_attn.sum(1).sort(dim=-1, descending=True)
                window_size = int((fg_src.shape[-1] * fg_src.shape[-2] // src[0].shape[0]) ** 0.5)
                fg_src_windows = window_partition(fg_src, window_size).view(bs, -1, window_size ** 2, c)
                fg_pos_windows = window_partition(fg_pos, window_size).view(bs, -1, window_size ** 2, c)

                fg_src_windows = torch.gather(fg_src_windows, 1,
                                              indexes[..., :n // self.omega_rate][..., None, None].expand(-1, -1,
                                                                                             window_size ** 2,
                                                                                             c)).view(bs,
                                                                                                      -1,
                                                                                                      c).permute(1, 0,
                                                                                                                 2)
                fg_pos_windows = torch.gather(fg_pos_windows, 1,
                                              indexes[..., :n // self.omega_rate][..., None, None].expand(-1, -1,
                                                                                             window_size ** 2,
                                                                                             c)).view(bs,
                                                                                                      -1,
                                                                                                      c).permute(1, 0,
                                                                                                                 2)

                # We do not use the dynamic positional queries in the sparse high-resolution branch
                output, _ = self.transformer_cross_attention_layers[i](
                    output, fg_src_windows,
                    memory_mask=None,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=fg_pos_windows, query_pos=query_embed_params
                )

            else:
                output, attn = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed,
                )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask, _ = self.forward_prediction_heads(output, mask_features,
                                                                                   attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            # Core code block for generating the dynamic positional queries
            attn = attn.flatten(2)
            if i % 4 != 3:
                query_embed = (attn.permute(1, 2, 0).unsqueeze(-1) * pos[i % self.num_feature_levels].unsqueeze(0)).sum(1)
                query_embed = self.adapt_pos2d(query_embed + query_embed_params)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = attn_mask.sigmoid().flatten(2)
        pooling_mask = (attn_mask < 0.5).int()

        # We release the previous threshold for generating attention mask in Mask2former to 0.15
        final_attn_mask = (attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.15).bool()
        final_attn_mask = final_attn_mask.detach()

        return outputs_class, outputs_mask, final_attn_mask, pooling_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


class MultiScaleMaskedTransformerDecoderFASeg_onlyDQ(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            mask_classification=True,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=10,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
            omega_rate=32
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        # self.pos_encoding = nn.Embedding(32 * 32, hidden_dim)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.hidden_dim = hidden_dim

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayerFASeg(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding, we add one extra high-resolution scale
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # projection layer for generate positional queries
        self.adapt_pos2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # The rate for selecting high-resolution areas
        self.omega_rate = omega_rate

    def forward(self, x, mask_features, mask=None, pos_list_2d=None):
        # x is a list of multi-scale feature
        # assert len(x) == self.num_feature_levels
        src = []
        pos = []
        pos_2d = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])

            # Use the same positional encoding in the pixel decoder
            pos_encoding = pos_list_2d[i]
            pos_2d.append(pos_encoding)

            pos.append(pos_encoding.flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        fg_src = self.input_proj[-1](x[-1]) + self.level_embed.weight[-1][None, :, None, None]
        fg_pos = pos_2d[-1]

        n, bs, c = src[0].shape

        # QxNxC
        query_embed_params = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, pooling_mask = self.forward_prediction_heads(output, mask_features,
                                                                               attn_mask_target_size=size_list[0])

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        # We generate dynamic positional queries for the first layer based on the predicted
        # positions of the first prediction head in Mask2former
        query_embed = (pooling_mask.permute(1, 2, 0).unsqueeze(-1) * pos[0].unsqueeze(0)).sum(1) / \
                     (pooling_mask.sum(-1).permute(1,0)[..., None] + 1e-6)

        query_embed = self.adapt_pos2d(query_embed + query_embed_params)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output, attn = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask, _ = self.forward_prediction_heads(output, mask_features,
                                                                                   attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            # Core code block for generating the dynamic positional queries
            attn = attn.flatten(2)

            query_embed = (attn.permute(1, 2, 0).unsqueeze(-1) * pos[i % self.num_feature_levels].unsqueeze(0)).sum(1)
            query_embed = self.adapt_pos2d(query_embed + query_embed_params)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = attn_mask.sigmoid().flatten(2)
        pooling_mask = (attn_mask < 0.5).int()

        # We release the previous threshold for generating attention mask in Mask2former to 0.15
        final_attn_mask = (attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.15).bool()
        final_attn_mask = final_attn_mask.detach()

        return outputs_class, outputs_mask, final_attn_mask, pooling_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
        
class MultiScaleMaskedTransformerDecoderFASeg_onlyDH(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            mask_classification=True,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=10,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
            omega_rate=32
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        # self.pos_encoding = nn.Embedding(32 * 32, hidden_dim)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.hidden_dim = hidden_dim

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayerFASeg(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding, we add one extra high-resolution scale
        self.num_feature_levels = 4
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # projection layer for generate positional queries
        self.adapt_pos2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # The rate for selecting high-resolution areas
        self.omega_rate = omega_rate

    def forward(self, x, mask_features, mask=None, pos_list_2d=None):
        # x is a list of multi-scale feature
        # assert len(x) == self.num_feature_levels
        src = []
        pos = []
        pos_2d = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])

            # Use the same positional encoding in the pixel decoder
            pos_encoding = pos_list_2d[i]
            pos_2d.append(pos_encoding)

            pos.append(pos_encoding.flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        fg_src = self.input_proj[-1](x[-1]) + self.level_embed.weight[-1][None, :, None, None]
        fg_pos = pos_2d[-1]

        n, bs, c = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, pooling_mask = self.forward_prediction_heads(output, mask_features,
                                                                               attn_mask_target_size=size_list[0])

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        # We generate dynamic positional queries for the first layer based on the predicted
        # positions of the first prediction head in Mask2former
        # query_embed = (pooling_mask.permute(1, 2, 0).unsqueeze(-1) * pos[0].unsqueeze(0)).sum(1) / \
        #              (pooling_mask.sum(-1).permute(1,0)[..., None] + 1e-6)

        # query_embed = self.adapt_pos2d(query_embed + query_embed_params)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first

            if i % 4 == 0:

                # Keep the attention in the first cross-attention layer
                output, attn = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed,
                )

                cg_attn = attn

            elif i % 4 == 3:

                # Select top-k areas in high-resolution attention maps
                _, indexes = cg_attn.sum(1).sort(dim=-1, descending=True)
                window_size = int((fg_src.shape[-1] * fg_src.shape[-2] // src[0].shape[0]) ** 0.5)
                fg_src_windows = window_partition(fg_src, window_size).view(bs, -1, window_size ** 2, c)
                fg_pos_windows = window_partition(fg_pos, window_size).view(bs, -1, window_size ** 2, c)

                fg_src_windows = torch.gather(fg_src_windows, 1,
                                              indexes[..., :n // self.omega_rate][..., None, None].expand(-1, -1,
                                                                                             window_size ** 2,
                                                                                             c)).view(bs,
                                                                                                      -1,
                                                                                                      c).permute(1, 0,
                                                                                                                 2)
                fg_pos_windows = torch.gather(fg_pos_windows, 1,
                                              indexes[..., :n // self.omega_rate][..., None, None].expand(-1, -1,
                                                                                             window_size ** 2,
                                                                                             c)).view(bs,
                                                                                                      -1,
                                                                                                      c).permute(1, 0,
                                                                                                                 2)

                # We do not use the dynamic positional queries in the sparse high-resolution branch
                output, _ = self.transformer_cross_attention_layers[i](
                    output, fg_src_windows,
                    memory_mask=None,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=fg_pos_windows, query_pos=query_embed
                )

            else:
                output, attn = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed,
                )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask, _ = self.forward_prediction_heads(output, mask_features,
                                                                                   attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            # Core code block for generating the dynamic positional queries
            # attn = attn.flatten(2)
            # if i % 4 != 3:
            #     query_embed = (attn.permute(1, 2, 0).unsqueeze(-1) * pos[i % self.num_feature_levels].unsqueeze(0)).sum(1)
            #     query_embed = self.adapt_pos2d(query_embed + query_embed_params)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = attn_mask.sigmoid().flatten(2)
        pooling_mask = (attn_mask < 0.5).int()

        # We release the previous threshold for generating attention mask in Mask2former to 0.15
        final_attn_mask = (attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.15).bool()
        final_attn_mask = final_attn_mask.detach()

        return outputs_class, outputs_mask, final_attn_mask, pooling_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

class MultiScaleMaskedTransformerDecoder_OurDH_v3(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            mask_classification=True,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=10,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
            omega_rate=32
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        # self.pos_encoding = nn.Embedding(32 * 32, hidden_dim)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.hidden_dim = hidden_dim

        for layer in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayerFASeg(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding, we add one extra high-resolution scale
        self.num_feature_levels = 4
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # projection layer for generate positional queries
        self.adapt_pos2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # The rate for selecting high-resolution areas
        self.omega_rate = omega_rate

    def forward(self, x, mask_features, mask=None, pos_list_2d=None):
        # x is a list of multi-scale feature
        # assert len(x) == self.num_feature_levels
        src = []
        pos = []
        pos_2d = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])

            # Use the same positional encoding in the pixel decoder
            pos_encoding = pos_list_2d[i]
            pos_2d.append(pos_encoding)

            pos.append(pos_encoding.flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        fg_src = self.input_proj[-1](x[-1]) + self.level_embed.weight[-1][None, :, None, None]
        fg_pos = pos_2d[-1]

        n, bs, c = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, pooling_mask = self.forward_prediction_heads(output, mask_features,
                                                                               attn_mask_target_size=size_list[0])

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        # We generate dynamic positional queries for the first layer based on the predicted
        # positions of the first prediction head in Mask2former
        # query_embed = (pooling_mask.permute(1, 2, 0).unsqueeze(-1) * pos[0].unsqueeze(0)).sum(1) / \
        #              (pooling_mask.sum(-1).permute(1,0)[..., None] + 1e-6)

        # query_embed = self.adapt_pos2d(query_embed + query_embed_params)            

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first

            # Keep the attention in the first cross-attention layer
            output, _ = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask, _ = self.forward_prediction_heads(output, mask_features,
                                                                                   attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            # Core code block for generating the dynamic positional queries
            # attn = attn.flatten(2)
            # if i % 4 != 3:
            #     query_embed = (attn.permute(1, 2, 0).unsqueeze(-1) * pos[i % self.num_feature_levels].unsqueeze(0)).sum(1)
            #     query_embed = self.adapt_pos2d(query_embed + query_embed_params)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = attn_mask.sigmoid().flatten(2)
        pooling_mask = (attn_mask < 0.5).int()

        # We release the previous threshold for generating attention mask in Mask2former to 0.15
        final_attn_mask = (attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.15).bool()
        final_attn_mask = final_attn_mask.detach()

        return outputs_class, outputs_mask, final_attn_mask, pooling_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
        
class CrossAttentionLayerRCDA_v4(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = DEACA_attention(d_model, nhead)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.adapt_pos1d = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def pos2posemb1d(self, pos, num_pos_feats=256, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., None] / dim_t
        posemb = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        return posemb

    def with_query_pos_embed(self, tensor, pos: Optional[Tensor], ax):
        if pos is None:
            return tensor
        tensor = tensor.permute(1,0,2)
        pos = pos.permute(1,0,2)
        pos = self.adapt_pos1d(self.pos2posemb1d(pos[..., ax]))
        return tensor + pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor], ax):
        if pos is None:
            return tensor
        b, e, h, w = tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]
        tensor = tensor.permute(0,2,3,1)
        pos = pos.permute(0,2,3,1)
        if ax == 0:
            pos = pos.mean(1).unsqueeze(1).repeat(1, h, 1, 1)
        else:
            pos = pos.mean(2).unsqueeze(2).repeat(1, 1, w, 1)
        return tensor + pos

    # query[5,4,256] memory[4,256,128,128]
    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        # We return the generated attention map
        tgt2 = self.multihead_attn(self.with_query_pos_embed(tgt, query_pos, 0),
                                self.with_query_pos_embed(tgt, query_pos, 1),
                                self.with_pos_embed(memory, pos, 0), 
                                self.with_pos_embed(memory, pos, 1),
                                memory.permute(0,2,3,1))

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(self.with_query_pos_embed(tgt, query_pos),
                                self.with_query_pos_embed(tgt, query_pos),
                                self.with_pos_embed(memory, pos, 0), 
                                self.with_pos_embed(memory, pos, 1),
                                memory.permute(0,2,3,1))
        
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)
    
class MultiScaleMaskedTransformerDecoder_OurDH_v4(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            mask_classification=True,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=10,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
            omega_rate=32
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        # self.pos_encoding = nn.Embedding(32 * 32, hidden_dim)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.hidden_dim = hidden_dim

        for layer in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            if (layer + 1) % 4 == 0:
                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayerRCDA_v4(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )
            else:
                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayerFASeg(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_embed = nn.Embedding(num_queries, 2)

        # level embedding, we add one extra high-resolution scale
        self.num_feature_levels = 4
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x, mask_features, mask=None, pos_list_2d=None):
        # x is a list of multi-scale feature
        # assert len(x) == self.num_feature_levels
        src = []
        pos = []
        pos_2d = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])

            # Use the same positional encoding in the pixel decoder
            pos_encoding = pos_list_2d[i]
            pos_2d.append(pos_encoding)

            pos.append(pos_encoding.flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        fg_src = self.input_proj[-1](x[-1]) + self.level_embed.weight[-1][None, :, None, None]
        fg_pos = pos_2d[-1]

        n, bs, c = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        query_pos_embed = self.query_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, pooling_mask = self.forward_prediction_heads(output, mask_features,
                                                                               attn_mask_target_size=size_list[0])

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        # We generate dynamic positional queries for the first layer based on the predicted
        # positions of the first prediction head in Mask2former
        # query_embed = (pooling_mask.permute(1, 2, 0).unsqueeze(-1) * pos[0].unsqueeze(0)).sum(1) / \
        #              (pooling_mask.sum(-1).permute(1,0)[..., None] + 1e-6)

        # query_embed = self.adapt_pos2d(query_embed + query_embed_params)            

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first

            if i % 4 == 3:
                output = self.transformer_cross_attention_layers[i](
                    output, fg_src,
                    memory_mask=None,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=fg_pos, query_pos=query_pos_embed)

            else:
                output, _ = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed,
                )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask, _ = self.forward_prediction_heads(output, mask_features,
                                                                                   attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = attn_mask.sigmoid().flatten(2)
        pooling_mask = (attn_mask < 0.5).int()

        # We release the previous threshold for generating attention mask in Mask2former to 0.15
        final_attn_mask = (attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.15).bool()
        final_attn_mask = final_attn_mask.detach()

        return outputs_class, outputs_mask, final_attn_mask, pooling_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


class CrossAttentionLayerRCDA_v5(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = rcda_rebuild(d_model, nhead)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.adapt_pos1d = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def pos2posemb1d(self, pos, num_pos_feats=256, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., None] / dim_t
        posemb = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        return posemb

    def with_query_pos_embed(self, tensor, pos: Optional[Tensor], ax):
        if pos is None:
            return tensor
        tensor = tensor.permute(1,0,2)
        pos = pos.permute(1,0,2)
        pos = self.adapt_pos1d(self.pos2posemb1d(pos[..., ax]))
        return tensor + pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor], ax):
        if pos is None:
            return tensor
        b, e, h, w = tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]
        tensor = tensor.permute(0,2,3,1)
        pos = pos.permute(0,2,3,1)
        if ax == 0:
            pos = pos.mean(1).unsqueeze(1).repeat(1, h, 1, 1)
        else:
            pos = pos.mean(2).unsqueeze(2).repeat(1, 1, w, 1)
        return tensor + pos

    # query[5,4,256] memory[4,256,128,128]
    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        # We return the generated attention map
        tgt2, attn = self.multihead_attn(self.with_query_pos_embed(tgt, query_pos, 0),
                                self.with_query_pos_embed(tgt, query_pos, 1),
                                self.with_pos_embed(memory, pos, 0), 
                                self.with_pos_embed(memory, pos, 1),
                                memory.permute(0,2,3,1))

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(self.with_query_pos_embed(tgt, query_pos),
                                self.with_query_pos_embed(tgt, query_pos),
                                self.with_pos_embed(memory, pos, 0), 
                                self.with_pos_embed(memory, pos, 1),
                                memory.permute(0,2,3,1))
        
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)
    
class MultiScaleMaskedTransformerDecoder_OurDH_v5(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            mask_classification=True,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=10,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
            omega_rate=32
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        # self.pos_encoding = nn.Embedding(32 * 32, hidden_dim)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.hidden_dim = hidden_dim

        for layer in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            if (layer + 1) % 4 == 0:
                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayerRCDA_v5(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )
            else:
                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayerFASeg(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_embed = nn.Embedding(num_queries, 2)

        # level embedding, we add one extra high-resolution scale
        self.num_feature_levels = 4
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x, mask_features, mask=None, pos_list_2d=None):
        # x is a list of multi-scale feature
        # assert len(x) == self.num_feature_levels
        src = []
        pos = []
        pos_2d = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])

            # Use the same positional encoding in the pixel decoder
            pos_encoding = pos_list_2d[i]
            pos_2d.append(pos_encoding)

            pos.append(pos_encoding.flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        fg_src = self.input_proj[-1](x[-1]) + self.level_embed.weight[-1][None, :, None, None]
        fg_pos = pos_2d[-1]

        n, bs, c = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        query_pos_embed = self.query_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, pooling_mask = self.forward_prediction_heads(0, output, mask_features,
                                                                               attn_mask_target_size=size_list[0])

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        # We generate dynamic positional queries for the first layer based on the predicted
        # positions of the first prediction head in Mask2former
        # query_embed = (pooling_mask.permute(1, 2, 0).unsqueeze(-1) * pos[0].unsqueeze(0)).sum(1) / \
        #              (pooling_mask.sum(-1).permute(1,0)[..., None] + 1e-6)

        # query_embed = self.adapt_pos2d(query_embed + query_embed_params)            

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first

            if i % 4 == 3:
                output, _ = self.transformer_cross_attention_layers[i](
                    output, fg_src,
                    memory_mask=None,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=fg_pos, query_pos=query_pos_embed)
                    
            else:
                output, attn = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed,
                )


            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask, _ = self.forward_prediction_heads(i, output, mask_features,
                                                                                   attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, idx, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        ###
        # base_dir = 'mask_rgb3'
        # if not os.path.exists(base_dir):
        #     os.makedirs(os.path.join(base_dir, 'fea'))
        #     os.makedirs(os.path.join(base_dir, 'query'))
        #     os.makedirs(os.path.join(base_dir, 'label'))
        # if idx == 12:
        #     for i in range(4):
        #         for j in range(100):
        #             dirname = os.path.join(base_dir, 'fea', '{}.png'.format(j))
        #             if os.path.exists(dirname):
        #                 continue
        #             data = mask_features[i]
        #             torch.save(data, dirname)
        #             break
        #     for i in range(4):
        #         for j in range(100):
        #             dirname = os.path.join(base_dir, 'query', '{}.png'.format(j))
        #             if os.path.exists(dirname):
        #                 continue
        #             data = mask_embed[i]
        #             torch.save(data, dirname)
        #             break

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = attn_mask.sigmoid().flatten(2)
        pooling_mask = (attn_mask < 0.5).int()

        # We release the previous threshold for generating attention mask in Mask2former to 0.15
        final_attn_mask = (attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.15).bool()
        final_attn_mask = final_attn_mask.detach()

        return outputs_class, outputs_mask, final_attn_mask, pooling_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]