import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple

from models.fusion import FusionBlock
from models.model_util import ConvLnAct, LayerNorm


# Self-Attention
class SA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = 1
        self.head_dim = dim
        # q, k, v 的线性变换层
        self.qkv = nn.Linear(dim, dim * 3)

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        # [batch_size, L, dim]
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # [batch_size, num_heads, L, dim]
        q = q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_t = k.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = v.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 计算注意力分数。[batch_size, num_heads, L, dim]
        scores = q @ k_t * 1.0 / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # [batch_size, C, H, W]
        output = (attn_weights @ v).permute(0, 2, 1, 3).reshape(B, L, C)
        output = output.permute(0, 2, 1).reshape(B, C, H, W)
        return output


class ELGCA(nn.Module):
    """
    Efficient local global context aggregation module
    dim: number of channels of input
    heads: number of heads utilized in computing attention
    """

    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.dwconv = nn.Conv2d(dim // 2, dim // 2, 3, padding=1, groups=dim // 2)
        self.qkvl = nn.Conv2d(dim // 2, (dim // 4) * self.heads, 1, padding=0)
        self.pool_q = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_k = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x1, x2 = torch.split(x, [C // 2, C // 2], dim=1)
        # apply depth-wise convolution on half channels
        x1 = self.act(self.dwconv(x1))

        # linear projection of other half before computing attention
        x2 = self.act(self.qkvl(x2))

        x2 = x2.reshape(B, self.heads, C // 4, H, W)

        # 1个头
        q = torch.sum(x2[:, :-3, :, :, :], dim=1)
        # q = torch.sum(x2[:, 0, :, :, :], dim=1)
        # 1个头
        # q = torch.sum(x2[:, 1, :, :, :], dim=1)
        k = x2[:, -3, :, :, :]

        q = self.pool_q(q)
        k = self.pool_k(k)
        # 1个头
        # q = torch.sum(x2[:, 2, :, :, :], dim=1)
        v = x2[:, -2, :, :, :].flatten(2)
        # 1个头
        # q = torch.sum(x2[:, 3, :, :, :], dim=1)
        l_feat = x2[:, -1, :, :, :]

        qk = torch.matmul(q.flatten(2), k.flatten(2).transpose(1, 2))
        qk = torch.softmax(qk, dim=1).transpose(1, 2)

        x2 = torch.matmul(qk, v).reshape(B, C // 4, H, W)

        x = torch.cat([x1, l_feat, x2], dim=1)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, heads=4, module='lgmm'):
        super().__init__()
        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.dca = ELGCA(dim, heads=heads) if module == 'elgca' else SA(dim, num_heads=1)
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


class EncoderAblation(nn.Module):
    def __init__(self, patch_size=3, in_chans=3, embed_dims=[32, 64, 128, 256],
                 mlp_ratios=[4, 4, 4, 4], heads=[4, 4, 4, 4], depths=[3, 3, 4, 3], module=''):
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
            self.block1.append(EncoderBlock(dim=embed_dims[0], heads=heads[0],
                                            mlp_ratio=mlp_ratios[0], module=module))

        # Stage-2 (x1/8 scale)=32x32
        self.block2 = nn.ModuleList()
        for i in range(depths[1]):
            self.block2.append(EncoderBlock(dim=embed_dims[1], heads=heads[1],
                                            mlp_ratio=mlp_ratios[1], module=module))

        # Stage-3 (x1/16 scale)=16x16
        self.block3 = nn.ModuleList()
        for i in range(depths[2]):
            self.block3.append(EncoderBlock(dim=embed_dims[2], heads=heads[2],
                                            mlp_ratio=mlp_ratios[2], module=module))

        # stage-4 (x1/32 scale)=8x8
        self.block4 = nn.ModuleList()
        for i in range(depths[3]):
            self.block4.append(EncoderBlock(dim=embed_dims[3], heads=heads[3],
                                            mlp_ratio=mlp_ratios[3], module=module))

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


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_channels=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))

    def forward(self, x):
        x = self.proj(x)
        return x


class ELGCDecoder(nn.Module):
    def __init__(self, in_channels=[32, 64, 128, 256], embedding_dim=64, output_nc=2,
                 use_convT=False, align_corners=True):
        super().__init__()

        # settings
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.fusion = nn.ModuleList([
            FusionBlock(dim, out_chs=embedding_dim)
            for dim in in_channels
        ])

        # linear fusion layer to combine mult-scale features of all stages
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim, kernel_size=1,
                      padding=0, stride=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        # Final predction head
        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2) \
                       if use_convT else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))

        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2) \
                       if use_convT else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))

        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.active = nn.Sigmoid()

    def forward(self, inputs1, inputs2):
        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = inputs1  # len=4, 1/4, 1/8, 1/16, 1/32
        c1_2, c2_2, c3_2, c4_2 = inputs2  # len=4, 1/4, 1/8, 1/16, 1/32

        x = [(i1, i2) for (i1, i2) in zip(inputs1, inputs2)]
        _c1, _c2, _c3, _c4 = [fusion(i1, i2) for (fusion, (i1, i2)) in zip(self.fusion, x)]
        n, _, h, w = c4_1.shape
        outputs = []
        # Stage 4: x1/32 scale
        # _c4_1 = self.linear_c4(c4_1)
        # _c4_2 = self.linear_c4(c4_2)
        # _c4 = self.diff_c4([_c4_1, _c4_2])
        _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        # _c3_1 = self.linear_c3(c3_1)
        # _c3_2 = self.linear_c3(c3_2)
        # _c3 = self.diff_c3([_c3_1, _c3_2])
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        # _c2_1 = self.linear_c2(c2_1)
        # _c2_2 = self.linear_c2(c2_2)
        # _c2 = self.diff_c2([_c2_1, _c2_2])
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        # _c1_1 = self.linear_c1(c1_1)
        # _c1_2 = self.linear_c1(c1_2)
        # _c1 = self.diff_c1([_c1_1, _c1_2])

        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat([_c4_up, _c3_up, _c2_up, _c1], dim=1))

        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        # Residual block
        x = self.dense_2x(x)
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)

        outputs.append(cp)

        return outputs


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(torch.nn.Module):
    """
    Transpose convolution layer to upsample the feature maps
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    """
    Residual convolutional block for feature enhancement in decoder
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class LinearProj(nn.Module):
    """
    Linear projection used to reduce the number of channels and feature mixing.
    input_dim: number of channels of input features
    embed_dim: number of channels for output features
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, embed_dim, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.proj(x)
        return x
