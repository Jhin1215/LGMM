import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # weight = self.weight.to(x.device)
            # bias = self.bias.to(x.device)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvLnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, bias=True,
                 act_func='gelu', data_format='channels_first', norm=''):
        super().__init__()
        if act_func.lower() == 'gelu':
            self.act_func = nn.GELU()
        elif act_func.lower() == 'relu':
            self.act_func = nn.ReLU()
        elif act_func.lower() == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func.lower() == 'softmax':
            self.act_func = nn.Softmax(dim=1)
        else:
            self.act_func = nn.Identity()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(dilation * (kernel_size - 1)) // 2,
                              dilation=dilation, groups=groups, bias=bias)
        if norm.lower() == 'ln':
            self.norm = LayerNorm(out_channels, data_format=data_format)
            nn.init.constant_(self.norm.weight, 1.0)
            nn.init.constant_(self.norm.bias, 0)
        elif norm.lower() == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm.lower() == 'gn':
            groups = max(1, min(32, out_channels // 16))
            if groups % out_channels !=0:
                groups = 1
            self.norm = nn.GroupNorm(groups, out_channels)
            nn.init.constant_(self.norm.weight, 1.0)
            nn.init.constant_(self.norm.bias, 0)
        else:
            self.norm = nn.Identity()
         
        # Depthwise 卷积标准初始化
        if groups == in_channels: 
            nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        else:
            # 手动计算GELU的初始化标准差
            fan = nn.init._calculate_correct_fan(self.conv.weight, mode='fan_in')
            std = math.sqrt(2.0) / math.sqrt(fan)  # 修正增益系数
            nn.init.normal_(self.conv.weight, 0, std)
            nn.init.constant_(self.conv.bias, 0.01)
            # 或者使用截断正态分布（效果更好）
            # nn.init.trunc_normal_(self.conv.weight, mean=0, std=std, a=-2*std, b=2*std)
            if out_channels == 2:
                nn.init.constant_(self.conv.bias, 0.0)
            else:
                nn.init.constant_(self.conv.bias, 0.1)

    def forward(self, x):
        return self.act_func(self.norm(self.conv(x)))
