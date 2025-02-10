import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from torchvision.models import VisionTransformer

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

from rscd.models.backbones.lamba_util.csms6s import SelectiveScanCuda
from rscd.models.backbones.lamba_util.utils import Scan_FB_S, Merge_FB_S, CrossMergeS, CrossScanS, \
                        local_scan_zero_ones, reverse_local_scan_zero_ones

from rscd.models.backbones.lamba_util.csms6s import flops_selective_scan_fn, flops_selective_scan_ref, selective_scan_flop_jit

def my_gumbel_softmax(logits, k):
    # 添加 Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    gumbel_logits = logits + gumbel_noise

    # 获取 top-k 的索引
    topk_indices = torch.topk(gumbel_logits, k=k, dim=-1).indices

    # 构造 top-k one-hot 分布
    topk_onehot = torch.zeros_like(logits)
    topk_onehot.scatter_(dim=-1, index=topk_indices, value=1.0)
    return topk_onehot

def window_expansion(x, H, W):
  # x [b, 1, 4, 1, 1]
    b, _, num_win = x.shape
    H1, W1 = int(H/4), int(W/4)
    num_win1 = int(num_win/4)

    x = x.reshape(b, 1, num_win1, num_win1, 1).squeeze(-1)
    x = F.interpolate(x, scale_factor=H1)

    return x


def window_partition(x, quad_size=2):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (B, num_windows, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    H_quad = H // quad_size
    W_quad = W // quad_size

    x = x.view(B, C, quad_size, H_quad, quad_size, W_quad)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(B, -1, H_quad, W_quad, C)  #.permute(0, 2, 1, 3, 4)
    return windows


def window_reverse(windows):
    """
    Args:
        windows: (B, C, num_windows, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, N, H, W, C)
    """
    B, N, H_l, W_l, C = windows.shape
    scale = int((N)**0.5)
    H = H_l * scale

    W = W_l * scale

    x = windows.permute(0, 4, 1, 2, 3)
    x = x.view(B, C, N // scale, N // scale, H_l, W_l)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
    return x

class Predictor(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        if len(x.shape) == 4:
            B, C, H, W = x.size()
            x_rs = x.reshape(B, C, -1).permute(0, 2, 1)
        else:
            B, N, C = x.size()
            H = int(N**0.5)
            x_rs = x
        x_rs = self.in_conv(x_rs)
        B, N, C = x_rs.size()

        window_scale = int(H//2)
        local_x = x_rs[:, :, :C // 2]
        global_x = x_rs[:, :, C // 2:].view(B, H, -1, C // 2).permute(0, 3, 1, 2)
        global_x_avg = F.adaptive_avg_pool2d(global_x,  (2, 2)) # [b, c, 2, 2]
        global_x_avg_concat = F.interpolate(global_x_avg, scale_factor=window_scale)
        global_x_avg_concat = global_x_avg_concat.view(B, C // 2, -1).permute(0, 2, 1).contiguous()

        x_rs = torch.cat([local_x, global_x_avg_concat], dim=-1)

        x_score = self.out_conv(x_rs)
        x_score_rs = x_score.permute(0, 2, 1).reshape(B, 2, H, -1)
        return x_score_rs



# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
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


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


def shift_size_generate(index=0, H=0):
    sz = int(H // 8)
    if (index%5)==1:
        shift_size = (sz, sz)
        reverse_size = (-sz, -sz)
    elif (index%5)==2:
        shift_size = (-sz, -sz)
        reverse_size = (sz, sz)
    elif (index % 5) == 3:
        shift_size = (sz, -sz)
        reverse_size = (-sz, sz)
    elif (index%5)== 4:
        shift_size = (-sz, sz)
        reverse_size = (sz, -sz)
    return shift_size, reverse_size


# support: v01-v05; v051d,v052d,v052dc;
# postfix: _onsigmoid,_onsoftmax,_ondwconv3,_onnone;_nozact,_noz;_oact;_no32;
# history support: v2,v3;v31d,v32d,v32dc;
class SS2Dv2:
    def __initv2__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            channel_divide = 1,
            stage_num = 0,
            depth_num =0,
            block_depth = 0,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_proj = int(ssm_ratio * d_model)
        self.channel_divide = int(channel_divide)
        d_inner = int((ssm_ratio * d_model)//channel_divide)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32 = False, #checkpostfix("_no32", forward_type)
        self.oact = False  # checkpostfix("_oact", forward_type)
        self.disable_z = True  # checkpostfix("_noz", forward_type)
        self.disable_z_act = False  # checkpostfix("_nozact", forward_type)
        self.out_norm_none = False
        self.out_norm_dwconv3 = False
        self.out_norm_softmax = False
        self.out_norm_sigmoid = False

        if self.out_norm_none:
            self.out_norm = nn.Identity()
        elif self.out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_proj, d_proj, kernel_size=3, padding=1, groups=d_proj, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif self.out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif self.out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
            self.out_norm = LayerNorm(d_proj * 2)

        # forward_type debug =======================================
        self.forward_core = partial(self.forward_core, force_fp32=True, no_einsum=True)
        #FORWARD_TYPES.get(forward_type, None)
        self.stage_num = stage_num
        self.depth_num = depth_num
        # self.block_index = (sum(block_depth[0:stage_num]) + depth_num)if stage_num>=1 else depth_num
        self.quad_flag = False
        self.shift_flag = False
        if self.stage_num == 0 or self.stage_num==1:
            k_group = 4  # 4
            self.score_predictor = Predictor(d_proj)
            self.quad_flag = True
            if self.depth_num % 5:
                self.shift_flag = True
        else:
            k_group = 4  # 4

        # in proj =======================================
        #d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model * 2, d_proj * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_proj * 2,
                out_channels=d_proj * 2,
                groups=d_proj * 2,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj


        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_proj * 2, d_model * 2, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)

    def forward_core(
            self,
            x: torch.Tensor = None,
            # ==============================
            to_dtype=True,  # True: final out to dtype
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
            # ==============================
            SelectiveScan=SelectiveScanCuda,
            CrossScan=CrossScanS,
            CrossMerge=CrossMergeS,
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            **kwargs,
    ):
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)


        B, D, H, W = x.shape
        _, N = A_logs.shape
        K, _, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, "mamba")

        if self.quad_flag:
            # score prediction+
            quad_size = int(2)
            quad_number = quad_size * quad_size
            xA, xB = x.split(x.shape[1] // 2, 1)
            score = self.score_predictor(xA - xB)
            if self.shift_flag:
                shift_size, reverse_size = shift_size_generate(self.depth_num, H)

                x = torch.roll(x, shifts=shift_size, dims=(2, 3))

            if H % quad_number != 0 or W % quad_number != 0:
                newH, newW = math.ceil(H / quad_number) * quad_number, math.ceil(W / quad_number) * quad_number
                diff_H, diff_W = newH - H, newW - W
                x = F.pad(x, (0, diff_H, 0, diff_W, 0, 0))
                score = F.pad(score, (0, diff_H, 0, diff_W, 0, 0))

                B, D, H, W = x.shape
                L = H * W
                diff_flag = True
            else:
                diff_flag = False

            ### quad_one_stage
            score_window = F.adaptive_avg_pool2d(score[:, 1, :, :], (4, 4)) # b, 1, 2, 2
            locality_decision = my_gumbel_softmax(score_window.view(B, 1, -1), k = 6)  # [b, 1, 4, 1, 1]

            locality = window_expansion(locality_decision, H=int(H), W=int(W))  # [b, 1, l]
            xs_zeros_ones = None
            len_zeros = []
            indices_zeros = []
            # num_zeros = []
            indices_ones = []
            num_ones = []
            for i in range(B):
                x_zeros, x_ones, sub_len_zeros, sub_indices_zeros, sub_indices_ones, sub_num_ones = local_scan_zero_ones(locality[i], x[i])
                len_zeros.append(sub_len_zeros)
                indices_zeros.append(sub_indices_zeros)
                # num_zeros.append(sub_num_zeros)
                indices_ones.append(sub_indices_ones)
                num_ones.append(sub_num_ones)
                x_zeros_ones = torch.cat([x_zeros, x_ones], dim=-1)
                if xs_zeros_ones is None:
                    xs_zeros_ones = x_zeros_ones.unsqueeze(0)
                else:
                    xs_zeros_ones = torch.cat([xs_zeros_ones, x_zeros_ones.unsqueeze(0)], dim=0)
            xs_1 = Scan_FB_S.apply(xs_zeros_ones)  # b, k, c, l 

            xs_zeros_ones_h = None
            len_zeros_h = []
            indices_zeros_h = []
            # num_zeros_h = []
            indices_ones_h = []
            num_ones_h = []
            for i in range(B):
                x_zeros_h, x_ones_h, sub_len_zeros_h, sub_indices_zeros_h, sub_indices_ones_h, sub_num_ones_h = local_scan_zero_ones(locality[i], x[i], h_scan=True)
                len_zeros_h.append(sub_len_zeros_h)
                indices_zeros_h.append(sub_indices_zeros_h)
                # num_zeros_h.append(sub_num_zeros_h)
                indices_ones_h.append(sub_indices_ones_h)
                num_ones_h.append(sub_num_ones_h)
                x_zeros_ones_h = torch.cat([x_zeros_h, x_ones_h], dim=-1)
                if xs_zeros_ones_h is None:
                    xs_zeros_ones_h = x_zeros_ones_h.unsqueeze(0)
                else:
                    xs_zeros_ones_h = torch.cat([xs_zeros_ones_h, x_zeros_ones_h.unsqueeze(0)], dim=0)
            xs_2 = Scan_FB_S.apply(xs_zeros_ones_h)  # b, k, c, l 

            xs = torch.cat([xs_1, xs_2], dim=1)
        else:
            xs = CrossScan.apply(x) 

        L = L * 2
        D = D // 2
        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                             bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, L)

        if self.quad_flag:
            y1 = Merge_FB_S.apply(ys[:, 0:2])  # BCL
            y2 = Merge_FB_S.apply(ys[:, 2:])  # BCL
            L = L // 2
            D = D * 2
            # for quad
            y = None
            for i in range(B):
                y_1 = reverse_local_scan_zero_ones(indices_zeros[i], indices_ones[i], num_ones[i], y1[i, ..., :len_zeros[i]], y1[i, ..., len_zeros[i]:])
                y_2 = reverse_local_scan_zero_ones(indices_zeros_h[i], indices_ones_h[i], num_ones_h[i], y2[i, ..., :len_zeros_h[i]], y2[i, ..., len_zeros_h[i]:], h_scan=True)
                sub_y = y_1 + y_2
                if y is None:
                    y = sub_y.unsqueeze(0)
                else:
                    y = torch.cat([y, sub_y.unsqueeze(0)], dim=0)

            if diff_flag:
                y = y.reshape(B, D, H, -1)
                y = y[:, :, 0:-diff_H, 0:-diff_W].contiguous()
                H, W = H - diff_H, W - diff_W
            else:
                y = y.view(B, D, H, -1)

            if self.shift_flag:
                y = torch.roll(y, shifts=reverse_size, dims=(2, 3))
        else:
            ys = ys.view(B, K, D, H, W * 2)
            y: torch.Tensor = CrossMerge.apply(ys) 
            L = L // 2
            D = D * 2
            y = y.view(B, -1, H, W)

        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x) # 384
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)

        out = self.dropout(self.out_proj(y))
        return out



class SS2D(nn.Module, mamba_init, SS2Dv2):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            channel_divide = 1,
            stage_num = 0,
            depth_num = 0,
            block_depth = 0,
            # ======================
            **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
            channel_divide =channel_divide,stage_num = stage_num,depth_num=depth_num, block_depth=block_depth,
        )
        self.__initv2__(**kwargs)
        return
