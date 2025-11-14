import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange, repeat
from pytorch_wavelets import DWTForward, DWTInverse
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

# LN
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# Upsample
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')
    return src

# DoubleConv
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# BilinearUP
class BilinearUP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BilinearUP, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
# ChannelMixing
class CA(nn.Module):
    def __init__(self, in_ch, reduction=4):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // reduction, in_ch, 1, bias=False)
        )

        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = self.mlp(avg_out) + self.mlp(max_out)
        return x * self.sigmod(out) + x

# VSSM
class VSSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,   ## 原来是3
        expand=2,  ## 原来是2
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
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
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)

        return out

# WGM
class WGM(nn.Module):
    def __init__(self, dim):
        super(WGM, self).__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(mode='zero', wave='haar')
        self.group = nn.Sequential(
            nn.Conv2d(dim, dim*2, kernel_size=1),
            nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2),
            nn.Conv2d(dim*2, dim, kernel_size=1)
        )
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x):
        b, c, h, w = x.shape
        yL, yH = self.dwt(x)

        y_L_new = yL[:, :, :int(h/2), :int(w/2)]
        y_LH = yH[0][:, :, 0, :int(h/2), :int(w/2)]
        y_HL = yH[0][:, :, 1, :int(h/2), :int(w/2)]
        y_HH = yH[0][:, :, 2, :int(h/2), :int(w/2)]

        yL = self.dwconv(y_L_new)
        y_LH_new = self.group(y_LH)
        y_HL_new = self.group(y_HL)
        y_HH_new = self.group(y_HH)

        yH = (torch.stack((y_LH_new, y_HL_new, y_HH_new), dim=2),)

        x = self.idwt((yL, yH))

        return self.dwconv(x)

# HRGN
class HRGN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(HRGN, self).__init__()

        self.dilconv1 = nn.Conv2d(in_dim, in_dim//2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dilconv2 = nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dilconv3 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=2, dilation=2)

        self.dwconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)

        self.outconv = nn.Conv2d(in_dim, out_dim, kernel_size=1)


    def forward(self, x):
        dx1 = self.dilconv1(x)
        dx2 = self.dilconv2(dx1)
        dx3 = self.dilconv3(torch.cat((dx1, dx2), dim=1))

        x1 = self.outconv(F.gelu(self.dwconv(x)) * (dx3 + x))

        return x1

# DFAM
class DFAM(nn.Module):
    def __init__(self, in_dim, out_dim, d_state):
        super(DFAM, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.block = nn.ModuleList([])
        self.block.append(nn.ModuleList([
                LayerNorm(in_dim, 'WithBias'),
                VSSM(d_model=in_dim, dropout=0, d_state=d_state),
                WGM(in_dim),
                HRGN(in_dim, out_dim)
                ])
            )
        

    def forward(self, x):
        for (Norm, VSSM, WGM, HRGN) in self.block:
            x1 = VSSM(Norm(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * WGM(x) + x
            x2 = HRGN(Norm(x1)) + self.conv(x1)

        return x2

# MEFF
class MEFF(nn.Module):
    def __init__(self, dim):
        super(MEFF, self).__init__()

        self.ca = CA(dim*2)
        self.conv_ca = nn.Conv2d(dim*2, dim, kernel_size=1)

        self.dilconv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dilconv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dilconv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.conv = nn.Conv2d(dim, dim, kernel_size=1)
    
    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        x_ca = self.conv_ca(self.ca(x))

        x_init = self.dwconv(x_ca)
        x1 = self.dilconv1(x_init)
        x2 = self.dilconv2(x_init)
        x3 = self.dilconv3(x_init)

        x = x1 + x2 + x3 + x_init
        att = torch.sigmoid(self.conv(x))

        out = x_ca * att + x_ca

        return out

# DFAwareNet
class DFAwareNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dim=32, d_state=16, deep_supervision=True, **kwargs):
        super(DFAwareNet, self).__init__()
        filters = [dim, dim * 2, dim * 4, dim * 8, dim * 16]
        self.deep_supervision = deep_supervision
        self.maxpools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(4)])

        self.E1 = DoubleConv(in_channels, filters[0])

        self.E2 = DFAM(filters[0], filters[1], d_state=d_state)

        self.E3 = DFAM(filters[1], filters[2], d_state=d_state)

        self.E4 = DFAM(filters[2], filters[3], d_state=d_state)

        self.E5 = DFAM(filters[3], filters[4], d_state=d_state)

        self.UP5 = BilinearUP(filters[4], filters[3])
        self.D5 = MEFF(filters[3])

        self.UP4 = BilinearUP(filters[3], filters[2])
        self.D4 = MEFF(filters[2])

        self.UP3 = BilinearUP(filters[2], filters[1])
        self.D3 = MEFF(filters[1])

        self.UP2 = BilinearUP(filters[1], filters[0])
        self.D2 = MEFF(filters[0])

        self.D1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)

        # DeepSupervision
        self.conv5 = nn.Conv2d(filters[4], 1, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(filters[3], 1, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(filters[2], 1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(filters[1], 1, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(filters[0], 1, kernel_size=3, stride=1, padding=1)

    
    def forward(self, img):
        E1 = self.E1(img)
        
        E2 = self.maxpools[0](E1)
        E2 = self.E2(E2)

        E3 = self.maxpools[1](E2)
        E3 = self.E3(E3)

        E4 = self.maxpools[2](E3)
        E4 = self.E4(E4)

        E5 = self.maxpools[3](E4)
        E5 = self.E5(E5)

        D5 = self.UP5(E5)
        D5 = self.D5(D5, E4)

        D4 = self.UP4(D5)
        D4 = self.D4(D4, E3)

        D3 = self.UP3(D4)
        D3 = self.D3(D3, E2)

        D2 = self.UP2(D3)
        D2 = self.D2(D2, E1)

        Out = self.D1(D2)

        # 深度监督
        d_s1 = self.conv1(D2)
        d_s2 = self.conv2(D3)
        d_s2 = _upsample_like(d_s2, d_s1)
        d_s3 = self.conv3(D4)
        d_s3 = _upsample_like(d_s3, d_s1)
        d_s4 = self.conv4(D5)
        d_s4 = _upsample_like(d_s4, d_s1)
        d_s5 = self.conv5(E5)
        d_s5 = _upsample_like(d_s5, d_s1)
        if self.deep_supervision:
            outs = [d_s1, d_s2, d_s3, d_s4, d_s5, Out]
        else:
            outs = Out

        return outs

# 示例用法
from thop import profile
model = DFAwareNet().cuda()
input = torch.randn(1, 3, 256, 256).cuda()
Flops, Params = profile(model, inputs=(input, ))
# 计算量
print('Flops: % .4fG' % (Flops / 1000000000))
# 参数量
print('Params参数量: % .4fM' % (Params / 1000000))