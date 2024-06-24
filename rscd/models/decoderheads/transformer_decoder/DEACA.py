import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from torch.nn.functional import linear, softmax
from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, bias=False,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        # self.bias = bias
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=bias))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

class Conv1d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, bias=False,
                 norm_cfg=dict(type='BN1d', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        # self.bias = bias
        self.add_module('c', nn.Conv1d(
            a, b, ks, stride, pad, dilation, groups, bias=bias))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

class DEACA_attention(torch.nn.Module):
    def __init__(self, dim, num_heads,
                 activation=nn.ReLU, ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scaling = float(head_dim) ** -0.5

        self.to_q_row = Conv1d_BN(dim, dim, 1)
        self.to_q_col = Conv1d_BN(dim, dim, 1)
        self.to_k_row = Conv2d_BN(dim, dim, 1)
        self.to_k_col = Conv2d_BN(dim, dim, 1)
        self.to_v = Conv2d_BN(dim, dim, 1)

        self.proj = torch.nn.Sequential(activation(), Conv1d_BN(
            dim, dim, bn_weight_init=0))
        self.proj_encode_row = torch.nn.Sequential(activation(), Conv1d_BN(
            dim, dim, bn_weight_init=0))

        self.proj_encode_column = torch.nn.Sequential(activation(), Conv1d_BN(
            dim, dim, bn_weight_init=0))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pwconv = Conv2d_BN(head_dim, head_dim, ks=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, q_row, q_col, k_row, k_col, v):  
        # q [4,5,256]
        # k [4,128,128,256]
        # v [4,128,128,256]
        _, tgt_len, _ = q_row.shape
        B, H, W, C = v.shape # [4,128,128,256]

        q_row = self.to_q_row(q_row.transpose(1, 2)) # [4,256,5]
        q_col = self.to_q_col(q_col.transpose(1, 2)) # [4,256,5]
        k_row = self.to_k_row(k_row.permute(0, 3, 1, 2)) # [4,256,128,128]
        k_col = self.to_k_col(k_col.permute(0, 3, 1, 2)) # [4,256,128,128]
        v = self.to_v(v.permute(0, 3, 1, 2)) # [4,256,128,128]

        q_row = q_row.permute(2, 0, 1) # [5,4,256]
        q_col = q_col.permute(2, 0, 1) # [5,4,256]
        k_row = k_row.mean(-1).permute(2, 0, 1) # [128,4,256]
        k_col = k_col.mean(-2).permute(2, 0, 1) # [128,4,256]

        q_row = q_row.contiguous().view(tgt_len, B * self.num_heads, self.head_dim).transpose(0, 1) # [32,5,32]
        q_col = q_col.contiguous().view(tgt_len, B * self.num_heads, self.head_dim).transpose(0, 1) # [32,5,32]

        k_row = k_row.contiguous().view(-1, B * self.num_heads, self.head_dim).transpose(0, 1) # [32,128,32]
        k_col = k_col.contiguous().view(-1, B * self.num_heads, self.head_dim).transpose(0, 1) # [32,128,32]

        v = v.contiguous().permute(1,2,0,3).reshape(H, W, B * self.num_heads, self.head_dim).permute(2,0,1,3) # [32,128,128,32]
        v_avg = self.sigmoid(self.pwconv(self.avg_pool(v.permute(0,3,1,2)))).squeeze(-1).permute(0,2,1) # [32, 1, 32]

        k_row = k_row * v_avg
        k_col = k_col * v_avg

        v_row = v.mean(2) # [32,128,32]
        v_col = v.mean(1) # [32,128,32]

        attn_row = torch.matmul(q_row, k_row.transpose(1, 2)) * self.scaling # [32,5,128]
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, v_row)  # [32,5,32]
        xx_row = self.proj_encode_row(xx_row.permute(0, 2, 1).reshape(B, self.dim, tgt_len)) # [4,256,5]
    
        attn_col = torch.matmul(q_col, k_col.transpose(1, 2)) * self.scaling
        attn_col = attn_col.softmax(dim=-1)
        xx_col = torch.matmul(attn_col, v_col)  # [32,5,32]
        xx_col = self.proj_encode_column(xx_col.permute(0, 2, 1).reshape(B, self.dim, tgt_len)) # [4,256,5]

        xx = xx_row.add(xx_col)
        xx = self.proj(xx)

        return xx.squeeze(-1).permute(2,0,1)
    

 
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
 

class DEACA_attention_v3(torch.nn.Module):
    # avg + max pool
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim = embed_dim // num_heads
        self.scaling = float(head_dim) ** -0.5
        
        self.in_proj_weight = Parameter(torch.empty(5 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(5 * embed_dim))
        self.proj_encode_row = Linear(embed_dim, embed_dim, bias=True)
        self.proj_encode_col = Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = torch.nn.Conv2d(head_dim, head_dim, 1)
        self.activate = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query_row, query_col, key_row, key_col, value):  
        bsz, tgt_len, embed_dim = query_row.size()

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = 0
        _end = embed_dim
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q_row = linear(query_row, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 1
        _end = embed_dim * 2
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q_col = linear(query_col, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 2
        _end = embed_dim * 3
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k_row = linear(key_row, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 3
        _end = embed_dim * 4
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k_col = linear(key_col, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 4
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        v = linear(value, _w, _b)
        
        # q [4,5,256]
        # k [4,128,128,256]
        # v [4,128,128,256]
        _k_row, _k_col = k_row, k_col
        _, tgt_len, _ = q_row.shape
        B, H, W, C = v.shape # [4,128,128,256]
        q_row = q_row.transpose(0, 1) # [5,4,256]
        q_col = q_col.transpose(0, 1) # [5,4,256]
        k_row = k_row.mean(1).transpose(0, 1) # [128,4,256]
        k_col = k_col.mean(2).transpose(0, 1) # [128,4,256]

        q_row = q_row.contiguous().view(tgt_len, B * self.num_heads, self.head_dim).transpose(0, 1) # [32,5,32]
        q_col = q_col.contiguous().view(tgt_len, B * self.num_heads, self.head_dim).transpose(0, 1) # [32,5,32]

        k_row = k_row.contiguous().view(-1, B * self.num_heads, self.head_dim).transpose(0, 1) # [32,128,32]
        k_col = k_col.contiguous().view(-1, B * self.num_heads, self.head_dim).transpose(0, 1) # [32,128,32]

        _k_row = _k_row.contiguous().permute(1,2,0,3).reshape(H, W, B * self.num_heads, self.head_dim).permute(2,0,1,3) # [32,128,128,32]
        _k_col = _k_col.contiguous().permute(1,2,0,3).reshape(H, W, B * self.num_heads, self.head_dim).permute(2,0,1,3) # [32,128,128,32]
        v = v.contiguous().permute(1,2,0,3).reshape(H, W, B * self.num_heads, self.head_dim).permute(2,0,1,3) # [32,128,128,32]
        v_avg = self.sigmoid(self.conv(self.avg_pool(v.permute(0,3,1,2)))).squeeze(-1).permute(0,2,1) # [32, 1, 32]

        k_row = k_row * v_avg # [32,128,32]
        k_col = k_col * v_avg # [32,128,32]

        v_row = v.mean(1)
        v_col = v.mean(2)

        attn_row = torch.matmul(q_row, k_row.transpose(1, 2)) * self.scaling # [32,5,128]
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, v_row)  # [32,5,32]
        xx_row = self.proj_encode_row(xx_row.permute(0, 2, 1).reshape(B, self.embed_dim, tgt_len).permute(2,0,1)) # [4,256,5]
    
        attn_col = torch.matmul(q_col, k_col.transpose(1, 2)) * self.scaling
        attn_col = attn_col.softmax(dim=-1)
        xx_col = torch.matmul(attn_col, v_col)  # [32,5,32]
        xx_col = self.proj_encode_col(xx_col.permute(0, 2, 1).reshape(B, self.embed_dim, tgt_len).permute(2,0,1)) # [4,256,5]

        xx = xx_row.add(xx_col)
        xx = self.out_proj(xx)

        return xx

# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction, in_channels),
#             nn.Sigmoid()
#         )
 
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return y

class rcda_rebuild(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.in_proj_weight = Parameter(torch.empty(5 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(5 * embed_dim))
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        # self.conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)
        self.lin = torch.nn.Linear(embed_dim, embed_dim)
        # self.activate = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        # self.senet = SEBlock(embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query_row, query_col, key_row, key_col, value):
        v_avg = self.sigmoid(self.lin(self.avg_pool(value.permute(0,3,1,2)).squeeze(-1).squeeze(-1))).unsqueeze(-1).permute(2,0,1)
        # v_avg = self.senet(value.permute(0,3,1,2)).squeeze(-1).permute(2,0,1)

        bsz, tgt_len, embed_dim = query_row.size()
        src_len_row = key_row.size()[2]
        src_len_col = key_col.size()[1]

        num_heads = self.num_heads
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = 0
        _end = embed_dim
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q_row = linear(query_row, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 1
        _end = embed_dim * 2
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q_col = linear(query_col, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 2
        _end = embed_dim * 3
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k_row = linear(key_row, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 3
        _end = embed_dim * 4
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k_col = linear(key_col, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 4
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        v = linear(value, _w, _b)

        q_row = q_row.transpose(0, 1)
        q_col = q_col.transpose(0, 1)
        k_row = k_row.mean(1).transpose(0, 1) * v_avg
        k_col = k_col.mean(2).transpose(0, 1) * v_avg

        q_row = q_row * scaling
        q_col = q_col * scaling

        q_row = q_row.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        q_col = q_col.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)

        k_row = k_row.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k_col = k_col.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().permute(1,2,0,3).reshape(src_len_col,src_len_row, bsz*num_heads, head_dim).permute(2,0,1,3)

        attn_output_weights_row = torch.bmm(q_row, k_row.transpose(1, 2))
        attn_output_weights_col = torch.bmm(q_col, k_col.transpose(1, 2))

        attn_output_weights_col = softmax(attn_output_weights_col, dim=-1)
        attn_output_weights_row = softmax(attn_output_weights_row, dim=-1)

        b_ein, q_ein, h_ein = attn_output_weights_col.shape
        b_ein, h_ein, w_ein, c_ein = v.shape
        attn_output_col = torch.matmul(attn_output_weights_col, v.reshape(b_ein, h_ein, w_ein * c_ein)).reshape(b_ein, q_ein, w_ein, c_ein)
        attn_output = torch.matmul(attn_output_weights_row[:, :, None, :], attn_output_col).squeeze(-2).permute(1, 0, 2).reshape(tgt_len, bsz, embed_dim)
        ### the following code base on einsum get the same results
        # attn_output_col = torch.einsum("bqh,bhwc->bqwc", attn_output_weights_col, v)
        # attn_output = torch.einsum("bqw,bqwc->qbc", attn_output_weights_row, attn_output_col).reshape(tgt_len, bsz,embed_dim)

        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output, torch.einsum("bqw,bqh->qbhw",attn_output_weights_row,attn_output_weights_col).reshape(tgt_len,bsz,num_heads,src_len_col,src_len_row).mean(2)

if __name__ == '__main__':
    dim = 256
    num_heads = 8
    act_layer = nn.ReLU
    attn = DEACA_attention(dim, num_heads=num_heads, activation=act_layer)
    # q [4,5,256]
    # k [4,128,128,256]
    # v [4,128,128,256]
    q_row = torch.randn((4,5,256))
    q_col = torch.randn((4,5,256))
    k_row = torch.randn((4,128,128,256))
    k_col = torch.randn((4,128,128,256))
    v = torch.randn((4,128,128,256))
    out = attn(q_row, q_col, k_row, k_col, v)
    print(out.shape)