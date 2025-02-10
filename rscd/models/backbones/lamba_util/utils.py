
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndimage

def local_scan_zero_ones(locality, x, h_scan=False):
    # 第一步：将 `local` 展平以便识别连通区域
    local_flat = locality.squeeze().cpu().numpy() 
    # labeled_zeros, num_zeros = ndimage.label(local_flat == 0)  # 标记 0 的连通区域
    labeled_ones, num_ones = ndimage.label(local_flat == 1)  # 标记 1 的连通区域

    # 第二步：提取连通区域的索引
    indices_zeros = torch.tensor(local_flat)
    indices_ones = torch.tensor(labeled_ones)
    # 第三步：为每个连通区域创建掩码
    components_zeros = []
    components_ones = []

    if h_scan:
        x.transpose_(-1, -2)
        indices_zeros.transpose_(-1, -2)
        indices_ones.transpose_(-1, -2)

    # for i in range(1, num_zeros + 1):
    #     mask = (indices_zeros == i)
    #     components_zeros.append(x[:,mask])  # 使用掩码从 y 中提取值
    
    mask = (indices_zeros == 0)
    components_zeros.append(x[:,mask])

    for i in range(1, num_ones + 1):
        mask = (indices_ones == i)
        components_ones.append(x[:,mask])  # 使用掩码从 y 中提取值

    # 第四步：将这些区域平铺（即按题目要求扫描）
    flattened_zeros = torch.cat(components_zeros, dim=-1) # 将所有 0 区域合并
    flattened_ones = torch.cat(components_ones, dim=-1)  # 将所有 1 区域合并

    return flattened_zeros, flattened_ones, flattened_zeros.shape[-1], indices_zeros == 0, indices_ones, num_ones

def reverse_local_scan_zero_ones(indices_zeros, indices_ones, num_ones, flattened_zeros, flattened_ones, h_scan=False):
    C, H, W = flattened_zeros.shape[0], indices_ones.shape[-2], indices_ones.shape[-1]
    local_restored = torch.zeros((C, H, W)).float().cuda(flattened_zeros.get_device()) # 创建一个与原始矩阵形状相同的零矩阵
    # 填充 0 区域
    # start_idx = 0
    # for i in range(1, num_zeros + 1):
    #     mask = (indices_zeros == i)
    #     local_restored[:, mask] = flattened_zeros[:, start_idx:start_idx + mask.sum()]
    #     start_idx += mask.sum()

    mask = indices_zeros
    local_restored[:, mask] = flattened_zeros

    # 填充 1 区域
    start_idx = 0
    for i in range(1, num_ones + 1):
        mask = (indices_ones == i)
        local_restored[:, mask] = flattened_ones[:, start_idx:start_idx + mask.sum()]
        start_idx += mask.sum()

    if h_scan:
        local_restored.transpose_(-1, -2)
   
    return local_restored


def merge_lists(list1, list2):
    list1, list2 = list1.unsqueeze(-1), list2.unsqueeze(-1)
    merged_list = torch.concat([list1, list2], -1)
    return merged_list

class Scan_FB_S(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, L = x.shape
        ctx.shape = (B, C // 2, L)
        x1, x2 = torch.split(x, C // 2, 1)
        xs1, xs2 = x1.new_empty((B, 2, C // 2, L)), x2.new_empty((B, 2, C // 2, L))

        xs1[:, 0] = x1
        xs1[:, 1] = x1.flip(-1)
        xs2[:, 0] = x2
        xs2[:, 1] = x2.flip(-1)
        xs = merge_lists(xs1, xs2).reshape(B, 2, C // 2, L * 2)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, L = ctx.shape
        ys = ys.view(B, 2, C, L, 2)
        ys1, ys2 = ys[..., 0], ys[..., 1]
        y1 = ys1[:, 0, :, :] + ys1[:, 1, :, :].flip(-1)
        y2 = ys2[:, 0, :, :] + ys2[:, 1, :, :].flip(-1)
        y = torch.concat([y1, y2], 1)
        return y.view(B, C * 2, L).contiguous()


class Merge_FB_S(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, C, L = ys.shape
        ctx.shape = (B, K, C, L)
        ys = ys.view(B, K, C, -1, 2)
        ys1, ys2 = ys[..., 0], ys[..., 1]
        y1 = ys1[:, 0, :, :] + ys1[:, 1, :, :].flip(-1)
        y2 = ys2[:, 0, :, :] + ys2[:, 1, :, :].flip(-1)
        y = torch.concat([y1, y2], 1)
        return y.contiguous()

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, K, C, L = ctx.shape
        x1, x2 = torch.split(x, C, 1)
        xs1, xs2 = x1.new_empty((B, K, C, L // 2)), x2.new_empty((B, K, C, L // 2))
        xs1[:, 0] = x1
        xs1[:, 1] = x1.flip(-1)
        xs2[:, 0] = x2
        xs2[:, 1] = x2.flip(-1)
        xs = merge_lists(xs1, xs2).reshape(B, K, C, L)
        return xs

class CrossScanS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C // 2, H, W)
        x1, x2 = torch.split(x, x.shape[1] // 2, 1)
        xs1, xs2 = x1.new_empty((B, 4, C // 2, H * W)), x2.new_empty((B, 4, C // 2, H * W))
        xs1[:, 0] = x1.flatten(2, 3)
        xs1[:, 1] = x1.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs1[:, 2:4] = torch.flip(xs1[:, 0:2], dims=[-1])
        xs2[:, 0] = x2.flatten(2, 3)
        xs2[:, 1] = x2.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs2[:, 2:4] = torch.flip(xs2[:, 0:2], dims=[-1])
        xs = merge_lists(xs1, xs2).reshape(B, 4, C // 2, H * W * 2)
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys.view(B, 4, C, L, 2)
        ys1, ys2 = ys[..., 0], ys[..., 1]
        ys1 = ys1[:, 0:2] + ys1[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        ys2 = ys2[:, 0:2] + ys2[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y1 = ys1[:, 0] + ys1[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y2 = ys2[:, 0] + ys2[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = torch.concat([y1, y2], 1)
        return y.view(B, -1, H, W)


class CrossMergeS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        W = W // 2
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1, 2)
        ys1, ys2 = ys[..., 0], ys[..., 1]
        ys1 = ys1[:, 0:2] + ys1[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        ys2 = ys2[:, 0:2] + ys2[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y1 = ys1[:, 0] + ys1[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y2 = ys2[:, 0] + ys2[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y = torch.concat([y1, y2], 1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        C = C // 2
        x1, x2 = torch.split(x, x.shape[1] // 2, 1)
        xs1, xs2 = x1.new_empty((B, 4, C, L)), x2.new_empty((B, 4, C, L))
        xs1[:, 0] = x1
        xs1[:, 1] = x1.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs1[:, 2:4] = torch.flip(xs1[:, 0:2], dims=[-1])
        xs2[:, 0] = x2
        xs2[:, 1] = x2.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs2[:, 2:4] = torch.flip(xs2[:, 0:2], dims=[-1])
        xs = merge_lists(xs1, xs2).reshape(B, 4, C, H, W * 2)
        return xs, None, None