import torch
import torch.nn as nn

from models.model_util import ConvLnAct, LayerNorm
from models.fusion import FusionBlock


class MobileNetV2Block(nn.Module):
    def __init__(self, in_chs, out_chs, expansion=3):
        super().__init__()
        self.conv = nn.Sequential(
            ConvLnAct(in_chs, in_chs * expansion, 1, act_func='gelu', norm='gn'),
            ConvLnAct(in_chs * expansion, in_chs * expansion, 3, groups=in_chs * expansion,
                      act_func='gelu', norm='gn'),
            ConvLnAct(in_chs * expansion, out_chs, 1, norm='gn')
        )
        self.residual = (in_chs == out_chs)

    def forward(self, x):
        if self.residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, in_channels=[32, 64, 128, 256], embedding_dim=64,
                 output_nc=2, mam_blocks=[1, 1, 2, 3, 2]):
        super().__init__()
        self.fusion = nn.ModuleList([
            FusionBlock(dim, out_chs=embedding_dim)
            for dim in in_channels
        ])

        # 8x8
        self.msdmm_x8 = PyramidMaskMamba(embedding_dim, in_channels[3], mam_block=mam_blocks[4], layer='8x8')
        self.up_sample_x8 = nn.Sequential(
            ConvLnAct(in_channels[3], in_channels[3] * 4, 1),
            nn.PixelShuffle(2),
            MobileNetV2Block(in_channels[3], embedding_dim)
        )

        # 16x16
        self.msdmm_x16 = PyramidMaskMamba(embedding_dim, in_channels[2], mam_block=mam_blocks[3], layer='16x16')
        self.up_sample_x16 = nn.Sequential(
            ConvLnAct(in_channels[2], in_channels[2] * 4, 1),
            nn.PixelShuffle(2),
            MobileNetV2Block(in_channels[2], embedding_dim)
        )

        # 32x32
        self.msdmm_x32 = PyramidMaskMamba(embedding_dim, in_channels[1], mam_block=mam_blocks[2], layer='32x32')
        self.up_sample_x32 = nn.Sequential(
            ConvLnAct(in_channels[1], in_channels[1] * 4, 1),
            nn.PixelShuffle(2),
            MobileNetV2Block(in_channels[1], embedding_dim)
        )

        # 64x64
        self.msdmm_x64 = PyramidMaskMamba(embedding_dim, in_channels[0], mam_block=mam_blocks[1], layer='64x64')
        self.up_sample_x64 = nn.Sequential(
            ConvLnAct(in_channels[0], in_channels[0] * 4, 1),
            nn.PixelShuffle(2),
            MobileNetV2Block(in_channels[0], embedding_dim)
        )

        # 128x128
        self.msdmm_x128 = PyramidMaskMamba(embedding_dim, in_channels[0], mam_block=mam_blocks[0], layer='64x64')
        self.up_sample_x128 = nn.Sequential(
            ConvLnAct(in_channels[0], in_channels[0] * 4, 1),
            nn.PixelShuffle(2),
            MobileNetV2Block(in_channels[0], embedding_dim)
        )

        self.predict_head = nn.Sequential(
            ConvLnAct(embedding_dim, output_nc, 1)
        )

    def forward(self, inputs1, inputs2):
        x = [(input1, input2) for input1, input2 in zip(inputs1, inputs2)]
        x = [fusion(input1, input2) for fusion, (input1, input2) in zip(self.fusion, x)]
        # x_64=(bs, 64, 64, 64)
        # x_32=(bs, 96, 32, 32)
        # x_16=(bs, 128, 16, 16)
        # x_8=(bs, 256, 8, 8)
        x_64, x_32, x_16, x_8 = x

        x8 = self.msdmm_x8(x_8)
        output1 = self.up_sample_x8(x8)

        x_16 = x_16 + output1
        x16 = self.msdmm_x16(x_16)
        output2 = self.up_sample_x16(x16)

        x_32 = x_32 + output2
        x32 = self.msdmm_x32(x_32)
        output3 = self.up_sample_x32(x32)

        x_64 = x_64 + output3
        x64 = self.msdmm_x64(x_64)
        output4 = self.up_sample_x64(x64)

        x_128 = output4
        x128 = self.msdmm_x128(x_128)
        output = self.up_sample_x128(x128)
        return [self.predict_head(out) for out in [output, output4, output3, output2, output1]]


class PyramidMaskMamba(nn.Module):
    def __init__(self, in_channels, out_channels, mam_block=1, layer='8x8'):
        super().__init__()
        self.blocks = mam_block
        self.proj = ConvLnAct(in_channels, out_channels, 1)

        state_dim = max(out_channels // 4, 32)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'pyra_conv': MultiScaleDilatedConv(out_channels, out_channels, layer=layer),
                'mam': MaskMamba(out_channels, state_dim)
            })
            for _ in range(mam_block)
        ])
        self.fuse = ConvLnAct(out_channels, out_channels, 1, act_func='gelu', norm='gn')

    def forward(self, x):
        x = self.proj(x)
        for block in self.blocks:
            x_pyra = block['pyra_conv'](x)
            x_mamba = block['mam'](x)
            x = self.fuse(x_pyra + x_mamba)
        return x


class MultiScaleDilatedConv(nn.Module):
    def __init__(self, in_chs, out_chs, layer):
        super().__init__()
        DILATEDCONV_SETTING = {
            '8x8': [1],
            '16x16': [1, 1],
            '32x32': [1, 2, 3],
            '64x64': [1, 2, 3, 4],
            '128x128': [1, 2, 3, 4, 5]
        }
        setting = DILATEDCONV_SETTING[layer]
        avg_dim = in_chs // len(setting)
        self.dim_list = [avg_dim for _ in setting]
        self.dim_list[-1] = in_chs - (len(self.dim_list) - 1) * avg_dim

        self.dilated_conv = nn.ModuleList([
            ConvLnAct(sub_dim, sub_dim, 3, dilation=dilation, groups=sub_dim, act_func='relu', norm='bn')
            for (dilation, sub_dim) in zip(setting, self.dim_list)
        ])

        self.conv1x1 = ConvLnAct(in_chs, out_chs, 1, act_func='gelu')

    def forward(self, x):
        x_list = x.split(self.dim_list, dim=1)
        out_list = [
            conv(dim)
            for (conv, dim) in zip(self.dilated_conv, x_list)
        ]
        return self.conv1x1(torch.cat(out_list, dim=1))


def make_mam_block(dim, state_dim, blocks):
    state_dim = 32 if state_dim <= 32 else state_dim
    mam_list = []
    for i in range(blocks):
        mam_list.append(MaskMamba(dim, state_dim=state_dim))
    return mam_list


class MaskMamba(nn.Module):
    def __init__(self, dim, state_dim=64, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.norm = LayerNorm(dim, data_format="channels_first")
        self.mgd_ss2d = MGDSS2D(dim, state_dim=state_dim)
        self.dw_conv1 = ConvLnAct(dim, dim, 3, groups=dim, act_func='gelu', bias=False)
        self.dw_conv2 = ConvLnAct(dim, dim, 3, groups=dim, act_func='gelu', bias=False)
        self.ffn = FFN(dim, int(dim * mlp_ratio))
        self.weights = nn.Parameter(1e-2 * torch.ones(4))

    def forward(self, x):
        x_copy = x
        x = self.weights[0] * x + (1 - self.weights[0]) * self.dw_conv1(x)

        x_prev = x
        x = self.mgd_ss2d(x)

        x = self.weights[1] * x_prev + (1 - self.weights[1]) * self.dw_conv2(x)

        x = self.norm(x)
        x = self.weights[2] * x + (1 - self.weights[2]) * self.ffn(x)
        out = self.weights[3] * x + (1 - self.weights[3]) * x_copy
        return out


# MGD-SS2D=mask guided directional selective scan 2d
class MGDSS2D(nn.Module):
    def __init__(self, dim, state_dim=64):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.directions = 2
        # 分别表示水平正向、垂直正向、水平反向、垂直反向
        self.dir_config = {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1),
        }

        self.base_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, state_dim * 3, 1, bias=False),
        )

        self.dir_convs = nn.ModuleList([
            self._build_directional_conv(dx, dy) for dx, dy in self.dir_config.values()
        ])

        self.A_params = nn.ParameterList([
            nn.Parameter(torch.randn(state_dim) * 0.01)
            for _ in range(self.directions)
        ])
        self.D = nn.Parameter(torch.linspace(0.5, 1.5, self.directions))

        self.dir_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, self.directions, 1),
            nn.Softmax(dim=1)
        )

        nn.init.normal_(self.base_proj[0].weight, mean=0, std=0.01)
        for param in self.A_params:
            nn.init.trunc_normal_(param, mean=0.5, std=0.2, a=0.1, b=0.9)
        nn.init.constant_(self.D, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        base_params = self.base_proj(x)

        dir_outputs = []
        for dir_idx in range(self.directions):
            flip_dims = []
            if self.dir_config[dir_idx][0] == -1:
                flip_dims.append(3)
            if self.dir_config[dir_idx][1] == -1:
                flip_dims.append(2)

            dir_input = base_params.clone()
            if flip_dims:
                dir_input = torch.flip(dir_input, dims=flip_dims)
            dir_params = self.dir_convs[dir_idx](dir_input)  # (B, 3*state_dim, H, W)

            B_param, C_param, dt = torch.split(
                dir_params,
                [self.state_dim] * 3,
                dim=1
            )
            B_param = B_param.flatten(2)  # (B, state_dim, H*W)
            C_param = C_param.flatten(2)  # (B, state_dim, H*W)
            dt = dt.flatten(2)  # (B, state_dim, H*W)
            A = (dt + self.A_params[dir_idx].view(1, -1, 1)).sigmoid()

            # (B, state_dim, H*W)
            AB = torch.softmax(A * B_param, dim=1)
            # (B, C, state_dim)
            # (B, C, H*W) @ (B, H*W, state_dim) → (B, C, state_dim)
            h = x.flatten(2) @ AB.transpose(1, 2)
            h = h / (x.size(2) * x.size(3)) ** 0.5
            h = torch.clamp(h, -50.0, 50.0)
            h = torch.softmax(h, dim=1)
            h = torch.clamp(h, -1e3, 1e3)
            # (B, C, state_dim) @ (B, state_dim, H*W) → (B, C, H*W)
            output_dir = h @ C_param
            output_dir = torch.softmax(output_dir, dim=1)
            if flip_dims:
                output_dir = torch.flip(output_dir, dims=flip_dims)

            output_dir = output_dir.view(B, C, H, W) * self.D[dir_idx]
            dir_outputs.append(output_dir)

        dir_weights = self.dir_attn(x)
        fused_output = torch.sum(
            torch.stack(dir_outputs, dim=1) * dir_weights.view(-1, self.directions, 1, 1, 1),
            dim=1
        )
        return fused_output

    def _build_directional_conv(self, dx, dy):
        conv = nn.Conv2d(3 * self.state_dim, 3 * self.state_dim, 3,
                         padding=1, groups=3 * self.state_dim)
        nn.init.normal_(conv.weight, mean=0, std=0.03)

        with torch.no_grad():
            kernel_template = torch.ones(3, 3)
            # horizontal scan
            if dx == 1 or dx == -1:
                kernel_template = torch.tensor([
                    [1.5, 1.2, 1.0],
                    [1.5, 1.2, 1.0],
                    [1.5, 1.2, 1.0]
                ], dtype=torch.float32)
            #
            elif dy == 1 or dy == -1:
                kernel_template = torch.tensor([
                    [1.5, 1.5, 1.5],
                    [1.2, 1.2, 1.2],
                    [1.0, 1.0, 1.0]
                ], dtype=torch.float32)

            for i in range(conv.weight.shape[0]):
                channel_multiplier = 0.8 + 0.4 * torch.rand(1)
                conv.weight.data[i] *= kernel_template * channel_multiplier

        return conv


class FFN(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.fc1 = ConvLnAct(in_dim, dim, 1, act_func='gelu')
        self.fc2 = ConvLnAct(dim, in_dim, 1)

    def forward(self, x):
        x = self.fc2(self.fc1(x))
        return x
