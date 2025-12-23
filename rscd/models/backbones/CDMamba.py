from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba as Mamba_ssm
from rscd.models.backbones.mamba_customer import ConvMamba, L_GF_Mamba, G_GL_Mamba


def get_conv_layer(spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False):
    if spatial_dims != 2:
        raise NotImplementedError("Only 2D is supported")
    padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


def get_norm_layer(name, spatial_dims, channels):
    if isinstance(name, (tuple, list)):
        norm_type = name[0].lower()
        args = name[1]
        if norm_type == "group":
            return nn.GroupNorm(num_groups=args.get("num_groups", 8), num_channels=channels)
    return nn.Identity()


def get_act_layer(act):
    if isinstance(act, (tuple, list)):
        act_name = act[0].lower()
        inplace = act[1].get("inplace", True)
        if act_name == "relu":
            return nn.ReLU(inplace=inplace)
    elif isinstance(act, str):
        if act.lower() == "silu":
            return nn.SiLU()
        if act.lower() == "relu":
            return nn.ReLU(inplace=True)
    return nn.ReLU()


def get_upsample_layer(spatial_dims, in_channels, upsample_mode="nontrainable"):
    return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


def get_dwconv_layer(
        spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
        bias: bool = False
):
    padding = kernel_size // 2
    depth_conv = nn.Conv2d(
        in_channels, in_channels, kernel_size=kernel_size, stride=stride, 
        padding=padding, groups=in_channels, bias=bias
    )
    point_conv = nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=1, 
        padding=0, groups=1, bias=bias
    )
    return torch.nn.Sequential(depth_conv, point_conv)


class SRCMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, conv_mode='deepwise'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.convmamba = ConvMamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v2",
            conv_mode=conv_mode
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.convmamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


def get_srcm_layer(
        spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1, conv_mode: str = "deepwise"
):
    srcm_layer = SRCMLayer(input_dim=in_channels, output_dim=out_channels, conv_mode=conv_mode)
    if stride != 1:
        if spatial_dims == 2:
            return nn.Sequential(srcm_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
    return srcm_layer


class SRCMBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            norm: tuple | str,
            kernel_size: int = 3,
            conv_mode: str = "deepwise",
            act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")
        
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_srcm_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, conv_mode=conv_mode
        )
        self.conv2 = get_srcm_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, conv_mode=conv_mode
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity
        return x


class L_GF(nn.Module):
    def __init__(self, dim, conv_mode="deepwise", resdiual=False, act="silu"):
        super(L_GF, self).__init__()
        self.fusionencoder = L_GF_Mamba(dim,bimamba_type="v2", conv_mode=conv_mode, act=act)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.resdiual = resdiual

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        id1 = x1
        id2 = x2
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        queryed_x1 = self.fusionencoder(x1, x2)
        queryed_x2 = self.fusionencoder(x2, x1)
        x1 = rearrange(queryed_x1, 'b (h w) c -> b c h w', h=h)
        x2 = rearrange(queryed_x2, 'b (h w) c -> b c h w', h=h)
        if self.resdiual:
            x1 = x1 + self.skip_scale*id1
            x2 = x2 + self.skip_scale*id2
        return x1, x2


class G_GF(nn.Module):
    def __init__(self, dim, resdiual=False, act="silu"):
        super(G_GF, self).__init__()
        self.fusionencoder = G_GL_Mamba(dim,bimamba_type="v2", act=act)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.resdiual = resdiual

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        id1 = x1
        id2 = x2
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        queryed_x1 = self.fusionencoder(x1, x2)
        queryed_x2 = self.fusionencoder(x2, x1)
        x1 = rearrange(queryed_x1, 'b (h w) c -> b c h w', h=h)
        x2 = rearrange(queryed_x2, 'b (h w) c -> b c h w', h=h)
        if self.resdiual:
            x1 = x1 + self.skip_scale*id1
            x2 = x2 + self.skip_scale*id2
        return x1, x2


class AdaptiveGate(nn.Module):
    def __init__(self, in_dim, num_expert=2):
        super().__init__()
        self.gate = nn.Linear(in_dim*2, num_expert, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_l, x_g):
        x_l = rearrange(x_l, 'b c h w -> b (h w) c')
        x_g = rearrange(x_g, 'b c h w -> b (h w) c')
        x_l = torch.mean(x_l, dim=1)
        x_g = torch.mean(x_g, dim=1)
        x_l_g = torch.cat([x_l, x_g], dim=-1)
        gate_score = self.gate(x_l_g)
        gate_score_n = self.softmax(gate_score)
        return gate_score_n


class CDMamba(nn.Module):
    def __init__(
            self,
            spatial_dims: int = 3,
            init_filters: int = 16,
            in_channels: int = 1,
            out_channels: int = 2,
            conv_mode: str = "deepwise",
            local_query_model = "orignal_dinner",
            dropout_prob: float | None = None,
            act: tuple | str = ("RELU", {"inplace": True}),
            norm: tuple | str = ("GROUP", {"num_groups": 8}),
            norm_name: str = "",
            num_groups: int = 8,
            use_conv_final: bool = True,
            blocks_down: tuple = (1, 2, 2, 4),
            blocks_up: tuple = (1, 1, 1),
            mode: str = "",
            up_mode="ResMamba",
            up_conv_mode="deepwise",
            resdiual=False,
            stage = 4,
            diff_abs="later", 
            mamba_act = "silu",
            upsample_mode: str = "nontrainable",
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")
        self.mode = mode
        self.stage = stage
        self.up_conv_mode = up_conv_mode
        self.mamba_act = mamba_act
        self.resdiual = resdiual
        self.up_mode = up_mode
        self.diff_abs = diff_abs
        self.conv_mode = conv_mode
        self.local_query_model = local_query_model
        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.channels_list = [self.init_filters, self.init_filters*2, self.init_filters*4, self.init_filters*8]
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        print(self.blocks_up)
        self.dropout_prob = dropout_prob
        self.act = act 
        self.act_mod = get_act_layer(act)
        
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        print(self.norm)
        
        self.upsample_mode = upsample_mode
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.srcm_encoder_layers = self._make_srcm_encoder_layers()
        self.srcm_decoder_layers, self.up_samples = self._make_srcm_decoder_layers(up_mode=self.up_mode)
        self.conv_final = self._make_final_conv(out_channels)


        self.l_gf1 = L_GF(self.channels_list[0], conv_mode=self.local_query_model, resdiual=self.resdiual, act=self.mamba_act)
        self.l_gf2 = L_GF(self.channels_list[1], conv_mode=self.local_query_model, resdiual=self.resdiual, act=self.mamba_act)
        self.l_gf3 = L_GF(self.channels_list[2], conv_mode=self.local_query_model, resdiual=self.resdiual, act=self.mamba_act)
        self.l_gf4 = L_GF(self.channels_list[3], conv_mode=self.local_query_model, resdiual=self.resdiual, act=self.mamba_act)
        self.l_gf = nn.Sequential(self.l_gf1, self.l_gf2, self.l_gf3, self.l_gf4)


        self.g_gf1 = G_GF(self.channels_list[0], resdiual=self.resdiual, act=self.mamba_act)
        self.g_gf2 = G_GF(self.channels_list[1], resdiual=self.resdiual, act=self.mamba_act)
        self.g_gf3 = G_GF(self.channels_list[2], resdiual=self.resdiual, act=self.mamba_act)
        self.g_gf4 = G_GF(self.channels_list[3], resdiual=self.resdiual, act=self.mamba_act)
        self.g_gf = nn.Sequential(self.g_gf1, self.g_gf2, self.g_gf3, self.g_gf4)

        self.ag1 = AdaptiveGate(self.channels_list[0])
        self.ag2 = AdaptiveGate(self.channels_list[1])
        self.ag3 = AdaptiveGate(self.channels_list[2])
        self.ag4 = AdaptiveGate(self.channels_list[3])
        self.ag = nn.Sequential(self.ag1, self.ag2, self.ag3, self.ag4)

        if dropout_prob is not None:
            self.dropout = nn.Dropout2d(dropout_prob)

    def _make_srcm_encoder_layers(self):
        srcm_encoder_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm, conv_mode = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm, self.conv_mode)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2 ** i
            downsample_mamba = (
                get_srcm_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2, conv_mode=conv_mode)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                downsample_mamba,
                *[SRCMBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act, conv_mode=conv_mode) for _ in range(item)]
            )
            srcm_encoder_layers.append(down_layer)
        return srcm_encoder_layers

    def _make_srcm_decoder_layers(self, up_mode):
        srcm_decoder_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        if up_mode == 'SRCM':
            Block_up = SRCMBlock
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            srcm_decoder_layers.append(
                nn.Sequential(
                    *[
                        Block_up(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act, conv_mode=self.up_conv_mode)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return srcm_decoder_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []

        for down in self.srcm_encoder_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.srcm_decoder_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)
        return x

    def forward(self, x1: torch.Tensor, x2:torch.Tensor) -> torch.Tensor:
        b, c, h, w = x1.shape
        x1, down_x1 = self.encode(x1)
        x2, down_x2 = self.encode(x2)
        down_x = []

        for i in range(len(down_x1)):
            x1, x2 = down_x1[i], down_x2[i]
            if self.diff_abs == "later":
                if self.mode == "AGLGF":
                    if i < self.stage:
                        x1_l, x2_l = self.l_gf[i](x1, x2)
                        x1_g, x2_g = self.g_gf[i](x1, x2)
                        x1_gate = self.ag[i](x1_l, x1_g)
                        x2_gate = self.ag[i](x2_l, x2_g)
                        x1 = x1_gate[:, 0:1].view(b, 1, 1, 1)*x1_l + x1_gate[:, 1:2].view(b, 1, 1, 1)*x1_g
                        x2 = x2_gate[:, 0:1].view(b, 1, 1, 1)*x2_l + x2_gate[:, 1:2].view(b, 1, 1, 1)*x2_g
                down_x.append(torch.abs(x1-x2))
        down_x.reverse()

        x = self.decode(down_x[0], down_x)
        return x

if __name__ == "__main__":
    device = "cuda:0"
    CDMamba = CDMamba(spatial_dims=2, in_channels=3, out_channels=2, init_filters=16, norm=("GROUP", {"num_groups": 8}),
                    mode="AGLGF", conv_mode='orignal', local_query_model="orignal_dinner",
                    stage=4, mamba_act="silu", up_mode="SRCM", up_conv_mode='deepwise', blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
                    resdiual=False, diff_abs="later").to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    y = CDMamba(x, x)
    print(y.shape)