import torch
import torch.nn as nn
from timm.layers import to_2tuple

from models.model_util import ConvLnAct, LayerNorm


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()
        self.fc1 = ConvLnAct(dim, dim * mlp_ratio, 1, act_func='gelu')
        self.pos = ConvLnAct(dim * mlp_ratio, dim * mlp_ratio, 3, groups=dim * mlp_ratio, act_func='gelu')
        self.fc2 = ConvLnAct(dim * mlp_ratio, dim, 1, act_func='gelu')

    def forward(self, x):
        x = self.fc1(x)
        x = x + self.pos(x)
        x = self.fc2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, heads, window_size):
        super().__init__()
        self.heads = heads
        self.window_size = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing='ij'
        )).flatten(1)

        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

    def forward(self, q, k, v, shift_size=0):
        if shift_size > 0:
            shifted_q = torch.roll(q, shifts=(-shift_size, -shift_size), dims=(2, 3))
        else:
            shifted_q = q

        q_windows, H, W = self.window_partition(shifted_q)  # [B*num, C, ws, ws]
        k_windows, _, _ = self.window_partition(k)
        v_windows, _, _ = self.window_partition(v)

        B_, C_, Wh, Ww = q_windows.shape
        # [B_, heads, tokens, dim_per_head]
        q = q_windows.view(B_, self.heads, C_ // self.heads, Wh * Ww).permute(0, 1, 3, 2)
        k_t = k_windows.view(B_, self.heads, C_ // self.heads, Wh * Ww).permute(0, 1, 2, 3)
        v = v_windows.view(B_, self.heads, C_ // self.heads, Wh * Ww).permute(0, 1, 3, 2)

        attn = (q @ k_t) * (1.0 / (C_ // self.heads) ** 0.5)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size, -1)  # [w^2, w^2, heads]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [heads, w^2, w^2]
        attn += relative_position_bias.unsqueeze(0)

        attn = torch.softmax(attn, dim=-1)
        x = (attn @ v).transpose(2, 3).reshape(B_, C_, Wh, Ww)
        x = self.window_reverse(x, H, W)
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))
        return x

    def window_partition(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.window_size, self.window_size,
                   W // self.window_size, self.window_size)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.window_size, self.window_size)
        return windows, H, W

    def window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / (H / self.window_size * W / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size,
                         windows.shape[1], self.window_size, self.window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
        return x


class ChannelsSoftSplit(nn.Module):
    def __init__(self, in_channels, num_branches, min_prob=0.25):
        super().__init__()
        self.num_branches = num_branches
        self.min_prob = min_prob
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvLnAct(in_channels, in_channels // 4, 1),
            ConvLnAct(in_channels // 4, num_branches, 1, act_func='softmax')
        )

    def forward(self, x):
        B, C, H, W = x.shape
        weights = self.ca(x)
        constructed_weights = self.min_prob + (1 - self.min_prob * self.num_branches) * weights
        return [x * constructed_weights[:, i].unsqueeze(dim=1) for i in range(self.num_branches)]


class MultiKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, layer=1):
        super().__init__()
        CONV_CONFIG = {
            1: {"k": [7, 5, 3, 1], "d": 1},
            2: {"k": [7, 5, 3, 1], "d": 1},
            3: {"k": [5, 3, 1], "d": 1},
            4: {"k": [3, 1], "d": 1}
        }
        config = CONV_CONFIG[layer]
        split_num = len(config["k"])
        in_base_ch = in_channels // split_num
        out_base_ch = out_channels // split_num

        self.in_splits = [in_base_ch] * (split_num - 1) + [in_channels - in_base_ch * (split_num - 1)]
        self.out_splits = [out_base_ch] * (split_num - 1) + [out_channels - out_base_ch * (split_num - 1)]

        self.dynamic_conv = nn.ModuleList()
        for i, ks in enumerate(config["k"]):
            self.dynamic_conv.append(
                ConvLnAct(self.in_splits[i], self.out_splits[i], ks,
                          dilation=config["d"], act_func='gelu', norm='ln')
                if ks <= 3 else
                nn.Sequential(
                    *(self._replace_big_conv_kernel(i, ks))
                )
            )

    def forward(self, x):
        in_splits = torch.split(x, self.in_splits, dim=1)
        out_list = [branch(in_split) for branch, in_split in zip(self.dynamic_conv, in_splits)]
        return torch.cat(out_list, dim=1)

    def _replace_big_conv_kernel(self, idx, ks):
        REPLACE_STRATEGY = {
            9: 4,
            7: 3,
            5: 2,
        }

        conv_list = []
        for i in range(REPLACE_STRATEGY[ks]):
            conv_list.append(
                ConvLnAct(self.in_splits[idx], self.in_splits[idx], 3, act_func='gelu', norm='ln')
                if i != REPLACE_STRATEGY[ks] - 1 else
                ConvLnAct(self.in_splits[idx], self.out_splits[idx], 3, act_func='gelu', norm='ln')
            )

        return conv_list


# DLGPE -> Dynamic Local Global Parallel Encoder
class DLGPE(nn.Module):
    def __init__(self, dim, heads=4, window_size=8, shift_size=3, layer=1):
        super().__init__()
        heads = 4 if layer <= 2 else 8
        self.window_attn = WindowAttention(heads, window_size)
        self.shift_size = shift_size
        self.channels_soft_split = ChannelsSoftSplit(in_channels=dim, num_branches=3)

        self.conv = MultiKernelConv(dim, dim, layer)

        self.norm = LayerNorm(dim, data_format='channels_first')
        self.w_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.sw_qkv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim),
            nn.Conv2d(dim, dim * 3, 1)
        )

        nn.init.xavier_normal_(self.sw_qkv[0].weight, gain=1e-4)
        nn.init.xavier_normal_(self.sw_qkv[1].weight, gain=1e-4)
        nn.init.xavier_normal_(self.sw_qkv[2].weight, gain=1e-4)

    def forward(self, x):
        x_conv, x_w, x_sw = self.channels_soft_split(x)

        x_w = self.w_qkv(x_w)
        q_w, k_w, v_w = x_w.chunk(3, dim=1)
        q_w = self.norm(q_w)
        x_w = self.window_attn(q_w, k_w, v_w)

        x_sw = self.sw_qkv(x_sw)
        q_sw, k_sw, v_sw = x_sw.chunk(3, dim=1)
        q_sw = self.norm(q_sw)
        x_sw = self.window_attn(q_sw, k_sw, v_sw, self.shift_size)

        x_conv = self.conv(x_conv)
        return x_conv + x_w + x_sw


class EncoderBlock(nn.Module):
    def __init__(self, dim, window_size, layer, mlp_ratio=4, heads=4):
        super(EncoderBlock, self).__init__()
        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.dca = DLGPE(dim, heads=heads, window_size=window_size, layer=layer)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio)

    def forward(self, x):
        x_copy = x
        x = self.layer_norm1(x)
        x = self.dca(x)
        # (bs, c, h, w)
        x = x_copy + x

        inp_copy = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = inp_copy + x
        return x


class Encoder(nn.Module):
    def __init__(self, patch_size=3, in_chans=3, embed_dims=[32, 64, 128, 256],
                 windows_list=[8, 8, 8, 4], num_classes=2, input_img_size=256,
                 mlp_ratios=[4, 4, 4, 4], drop_path_rate=0.,
                 heads=[4, 4, 4, 4], depths=[3, 3, 4, 3]):
        super().__init__()
        self.depths = depths
        self.embed_dims = embed_dims

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_channels=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_channels=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_channels=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_channels=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # Stage-1 (x1/4 scale)=64x64
        self.block1 = nn.ModuleList()
        for i in range(depths[0]):
            self.block1.append(EncoderBlock(dim=embed_dims[0], window_size=windows_list[0], heads=heads[0],
                                            mlp_ratio=mlp_ratios[0], layer=1))

        # Stage-2 (x1/8 scale)=32x32
        self.block2 = nn.ModuleList()
        for i in range(depths[1]):
            self.block2.append(EncoderBlock(dim=embed_dims[1], window_size=windows_list[1], heads=heads[1],
                                            mlp_ratio=mlp_ratios[1], layer=2))

        # Stage-3 (x1/16 scale)=16x16
        self.block3 = nn.ModuleList()
        for i in range(depths[2]):
            self.block3.append(EncoderBlock(dim=embed_dims[2], window_size=windows_list[2], heads=heads[2],
                                            mlp_ratio=mlp_ratios[2], layer=3))

        # stage-4 (x1/32 scale)=8x8
        self.block4 = nn.ModuleList()
        for i in range(depths[3]):
            self.block4.append(EncoderBlock(dim=embed_dims[3], window_size=windows_list[3], heads=heads[3],
                                            mlp_ratio=mlp_ratios[3], layer=4))

    def forward_features(self, x):
        outs = []
        # stage 1 -> 1/4
        x_embed = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x_embed = blk(x_embed)
        outs.append(x_embed)

        # stage 2 -> 1/8
        x_embed = self.patch_embed2(x_embed)
        for i, blk in enumerate(self.block2):
            x_embed = blk(x_embed)
        outs.append(x_embed)

        # stage 3 -> 1/16
        x_embed = self.patch_embed3(x_embed)
        for i, blk in enumerate(self.block3):
            x_embed = blk(x_embed)
        outs.append(x_embed)

        # stage 4 -> 1/32
        x_embed = self.patch_embed4(x_embed)
        for i, blk in enumerate(self.block4):
            x_embed = blk(x_embed)
        outs.append(x_embed)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_channels=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))

    def forward(self, x):
        x = self.proj(x)
        return x
