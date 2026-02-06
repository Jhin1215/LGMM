import torch
import torch.nn as nn

from models.model_util import LayerNorm, ConvLnAct


class FusionBlock(nn.Module):
    def __init__(self, dim, out_chs):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.feature_align = nn.Sequential(
            ConvLnAct(dim * 2, dim * 4, 1, act_func='gelu', norm='gn'),
            ConvLnAct(dim * 4, dim * 4, 3, groups=dim, act_func='gelu', norm='gn'),
            SpatioTemporalAttention(dim * 4),
            ConvLnAct(dim * 4, dim, 1, act_func='gelu', norm='gn'),
        )

        self.conv_out = nn.Sequential(
            ConvLnAct(dim, dim * 4, 1, act_func='gelu', norm='gn'),
            ConvLnAct(dim * 4, dim * 4, 3, groups=dim, act_func='gelu', norm='gn'),
            ConvLnAct(dim * 4, out_chs, 1, norm='gn'),
        )
        self.fusion_weight = nn.Parameter(0.5 * torch.ones(1))

    def forward(self, pre_imgs, post_imgs):
        pre_imgs = self.norm(pre_imgs)
        post_imgs = self.norm(post_imgs)
        combined = self.feature_align(torch.cat([pre_imgs, post_imgs], dim=1))

        # 可学习的空间翻转融合
        # combined = (combined + torch.flip(combined, dims=[2, 3])) * torch.sigmoid(self.fusion_weight)
        # combined = combined + torch.flip(combined, dims=[2, 3]) * torch.sigmoid(self.fusion_weight)
        return self.conv_out(combined)


class SpatioTemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 空间分支：捕获局部空间变化
        self.spatial_att = ConvLnAct(2, 1, 7, act_func='sigmoid')

        # 时相分支：捕获双时相依赖
        self.temporal_att = nn.Sequential(
            ConvLnAct(dim, dim, 3, groups=dim, act_func='gelu'),
            ConvLnAct(dim, dim // 2, 1, act_func='gelu'),
            ConvLnAct(dim // 2, dim, 1, act_func='sigmoid')
        )

    def forward(self, x):
        # 空间注意力计算
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_weight = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))  # [B,1,H,W]
        x = x + x * spatial_weight

        # 时相注意力计算
        temporal_weight = self.temporal_att(x)  # [B,C,H,W]
        x = x + x * temporal_weight
        return x
