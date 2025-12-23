
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union




@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)






class MambaBlock_CD(nn.Module):
    def __init__(self, d_model,d_conv, d_state, bias = True, conv_bias = True):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        # self.args = args


        self.norm = RMSNorm(d_model=d_model)


        self.d_inner = 2 * d_model
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.mlp_1 = nn.Linear(self.d_inner, d_model)
        self.mlp_2 = nn.Linear(self.d_inner, d_model)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D_p = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        

    def forward(self, t1,t2):

        ee1 = t1 
        ee2 = t2
        
        (b, l, d) = t1.shape
        t1 = self.norm(t1)
        
        t1_and_res = self.in_proj(t1)  # shape (b, l, 2 * d_in)
        (t1, res1) = t1_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        t1 = rearrange(t1, 'b l d_in -> b d_in l')
        t1 = self.conv1d(t1)[:, :, :l]
        t1 = rearrange(t1, 'b d_in l -> b l d_in')
        
        t1 = F.silu(t1)


        (b, l, d) = t2.shape
        t2 = self.norm(t2)
        
        t2_and_res = self.in_proj(t2)  # shape (b, l, 2 * d_in)
        (t2, res2) = t2_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        t2 = rearrange(t2, 'b l d_in -> b d_in l')
        t2 = self.conv1d(t2)[:, :, :l]
        t2 = rearrange(t2, 'b d_in l -> b l d_in')
        
        t2 = F.silu(t2)

        y1,y2 = self.cssm(t1,t2)
        
        y1 = y1 * F.silu(res1)
        y2 = y2 * F.silu(res2)
        
        output1 = self.out_proj(y1)
        output2 = self.out_proj(y2)



        return output1 + ee1, output2 + ee2

    
    def cssm(self, t1, t2):

        (d_in, n) = self.A_log.shape

        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        t1_dbl = self.x_proj(t1)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = t1_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)


        A_prim = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D_prim = self.D_p.float()

        t2_dbl = self.x_proj(t2)  # (b, l, dt_rank + 2*n)
        
        (delta, B_prim, C_prim) = t2_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(t1,t2, delta, A, B, C, D, A_prim, B_prim, C_prim, D_prim)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, t1,t2, delta, A, B, C, D, A_prim, B_prim, C_prim, D_prim):

        (b, l, d_in) = t1.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, t1, 'b l d_in, b l n, b l d_in -> b l d_in n')
        deltaB_u_prim = einsum(delta, B_prim, t2, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + torch.abs(deltaB_u[:, i] - deltaB_u_prim[:,i])
            y1 = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y1)
        y1 = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y1 = y1 + t1 * D


        (b, l, d_in) = t2.shape
        n = A_prim.shape[1]

        deltaA_prim = torch.exp(einsum(delta, A_prim, 'b l d_in, d_in n -> b l d_in n'))
        # deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA_prim[:, i] * x + torch.abs(deltaB_u[:, i] - deltaB_u_prim[:,i])
            y2 = einsum(x, C_prim[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y2)
        y2 = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y2 = y2 + t2 * D_prim
    
        return y1 ,y2
    




class MambaCSSM(nn.Module):

    def __init__(self, num_layers, d_model,d_conv, d_state, bias = True, conv_bias = True ):
        super().__init__()

        self.layers =  nn.ModuleList([MambaBlock_CD(d_model,d_conv, d_state, bias = True, conv_bias = True) for _ in range(num_layers)])


    def forward(self, t1,t2):

        for layer in self.layers:
            t1,t2 = layer(t1,t2)

        return t1,t2 

            





class MambaBlock(nn.Module):
    def __init__(self, d_model,d_conv, d_state, bias = True, conv_bias = True):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        # self.args = args


        self.d_inner = 2 * d_model
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output