# rsm_scan.py
import math
import torch
import torch.nn as nn


try:
    import selective_scan_cuda_oflex  # RS-Mamba oflex 核
    _HAVE_OFLEX = True
except Exception:
    _HAVE_OFLEX = False

try:
    import selective_scan_cuda_core   # 新 core 核
    _HAVE_CORE = True
except Exception:
    _HAVE_CORE = False

try:
    import selective_scan_cuda        # 旧 mamba 核
    _HAVE_MAMBA = True
except Exception:
    _HAVE_MAMBA = False


# =========================
# 2) 四个 SelectiveScan * 后端
# =========================
class SelectiveScanMamba(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None,
                delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus, False
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None,
                delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        if _HAVE_CORE:
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        else:
            # 退化到旧核（若存在）
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if _HAVE_CORE:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus
            )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None,
                delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanFake(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None,
                delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        x = delta
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return u

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        du = torch.zeros_like(u)
        ddelta = torch.zeros_like(delta)
        dA = torch.zeros_like(A)
        dB = torch.zeros_like(B)
        dC = torch.zeros_like(C)
        dD = torch.zeros_like(C if D is None else D)
        ddelta_bias = torch.zeros_like(C if delta_bias is None else delta_bias)
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


# =========================
# 3) 与 RSM 相同的 gather/scatter
# =========================
def antidiagonal_gather(tensor):
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)
    index = (torch.arange(W, device=tensor.device) - shift) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)

def diagonal_gather(tensor):
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)
    index = (shift + torch.arange(W, device=tensor.device)) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)

def diagonal_scatter(tensor_flat, original_shape):
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

def antidiagonal_scatter(tensor_flat, original_shape):
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


# =========================
# 4) 与 RSM 一致的 CrossScan / CrossMerge
# =========================
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 8, C, H * W))
        xs[:, 0] = x.flatten(2, 3)                        # 水平
        xs[:, 1] = x.transpose(2, 3).flatten(2, 3)        # 垂直
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])    # 反向
        xs[:, 4] = diagonal_gather(x)                     # 主对角
        xs[:, 5] = antidiagonal_gather(x)                 # 反对角
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])    # 对角反向
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(2, 3).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        y_da = diagonal_scatter(y_da[:, 0], (B, C, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, C, H, W))
        return (y_rb + y_da)

class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)

        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(2, 3).contiguous().view(B, D, -1)
        y_rb = y_rb.view(B, -1, H, W)

        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        y_da = diagonal_scatter(y_da[:, 0], (B, D, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, D, H, W))

        y_res = y_rb + y_da
        return y_res.view(B, D, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 8, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(2, 3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        x_img = x.view(B, C, H, W)
        xs[:, 4] = diagonal_gather(x_img)
        xs[:, 5] = antidiagonal_gather(x_img)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])
        return xs.view(B, 8, C, H, W)


# =========================
# 5) 对外接口：选择后端 & 8 向扫描
# =========================
def resolve_selective_scan():
    """
    与 RSM 一致的“按已导入模块”选择后端：
    优先级：Oflex > Core > Mamba > Fake
    """
    if _HAVE_OFLEX:
        return SelectiveScanOflex
    if _HAVE_CORE:
        return SelectiveScanCore
    if _HAVE_MAMBA:
        return SelectiveScanMamba
    return SelectiveScanFake


def eight_dir_rsm_scan(
    x: torch.Tensor,                    # (B, C, H, W)
    x_proj_weight: torch.Tensor,        # (K=8, R+2N, C)
    dt_projs_weight: torch.Tensor,      # (K=8, C, R)
    dt_projs_bias: torch.Tensor,        # (K=8, C)
    A_logs: torch.Tensor,               # (C, N)
    Ds: torch.Tensor,                   # (K*C,)
    SelectiveScan,                      # 传入上面 resolve_selective_scan() 的返回
    out_norm: nn.Module,                # 归一化/激活头（channels_first 时就是 LayerNorm(CF) 或 ConvDW）
    out_norm_shape: str = "v1",         # "v1": (B,C,H,W)；"v0": (B,H,W,C)
    delta_softplus: bool = True,
    force_fp32: bool = False,
):
    B, C, H, W = x.shape
    K, CP, C_ = x_proj_weight.shape
    assert K == 8 and C_ == C, "x_proj_weight 形状不匹配"
    N = A_logs.shape[1]
    R = dt_projs_weight.shape[2]
    L = H * W

    # 1) 8 向扫描
    xs = CrossScan.apply(x)  # (B, 8, C, L)

    # 2) 投影得到 dts/B/C
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float)).repeat(8, 1, 1).flatten(0, 1)           # (C, N)
    Bs = Bs.contiguous()                              # (B, K, N, L)
    Cs = Cs.contiguous()                              # (B, K, N, L)
    Ds = Ds.to(torch.float)                           # (K*C,)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.float(); dts = dts.float(); Bs = Bs.float(); Cs = Cs.float()

    def _ss(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        # 与 RSM 用法对齐：nrows/backnrows=1，oflex=True（参数占位）
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, 1, True)

    # 3) selective scan
    ys = _ss(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus).view(B, K, -1, H, W)

    # 4) 融合 & 还原
    y = CrossMerge.apply(ys)  # (B, C, L)

    # 5) 输出形状
    if out_norm_shape == "v1":  # (B,C,H,W)
        y = out_norm(y.view(B, C, H, W))
        return y
    else:                       # (B,H,W,C)
        y = y.transpose(1, 2).contiguous().view(B, H, W, C)
        y = out_norm(y)
        return y
