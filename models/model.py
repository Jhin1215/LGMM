import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder
from models.decoder import Decoder
from ablation.ablation_module import ELGCDecoder, EncoderAblation
from ablation.resnet import AdaptedResNet18
from ablation.encoder import Encoder as EncAblation
from ablation.decoder import Decoder as DecAblation


# LGMM-Net：A Local-Global Encoder and Mask Mamba Decoder Hybrid Network for Remote Sensing Change Detection
class LGMMNet(nn.Module):
    def __init__(self, in_channels=3, out_channel=1, depths=[3, 3, 4, 3], heads=[4, 4, 4, 4],
                 enc_channels=[64, 96, 128, 256], dec_embed_dim=256, ablation_setting=None):
        super().__init__()
        self.drop_path_rate = 0.1
        self.setting = ablation_setting
        # 通道数调优
        base_chs = 64
        enc_channels = [base_chs, int(base_chs * 1.5), base_chs * 2, base_chs * 4]
        dec_embed_dim = base_chs * 4

        # shared encoder
        if ablation_setting['encoder_module'] == 'lgmm':
            self.enc = Encoder(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                               mlp_ratios=[4, 4, 4, 4], drop_path_rate=self.drop_path_rate, depths=depths)
        elif ablation_setting['encoder_module'] == 'msa':
            self.enc = EncoderAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                       mlp_ratios=[4, 4, 4, 4], depths=depths, module='msa')
        elif ablation_setting['encoder_module'] == 'resnet18':
            self.enc = AdaptedResNet18(pre_trained=True, output_channels=enc_channels)
        elif ablation_setting['encoder_module'] == 'elgca':
            self.enc = EncoderAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                       mlp_ratios=[4, 4, 4, 4], depths=depths, module='elgca')
        elif ablation_setting['encoder_module'] == 'lgmm_only_multi_conv':
            setting = {'multi_conv': True,
                       'w_attn': False,
                       'sw_attn': False,
                       'num_branches': 1,
                       'order': 1,
                       'conv_config': None,
                       'w_linear': False,
                       'sw_linear': True,
                       }
            self.enc = EncAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                   mlp_ratios=[4, 4, 4, 4], depths=depths, setting=setting)
        elif ablation_setting['encoder_module'] == 'lgmm_only_w_attn':
            setting = {'multi_conv': False,
                       'w_attn': True,
                       'sw_attn': False,
                       'num_branches': 1,
                       'order': 2,
                       'conv_config': None,
                       'w_linear': False,
                       'sw_linear': True,
                       }
            self.enc = EncAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                   mlp_ratios=[4, 4, 4, 4], depths=depths, setting=setting)
        elif ablation_setting['encoder_module'] == 'lgmm_only_sw_attn':
            setting = {'multi_conv': False,
                       'w_attn': False,
                       'sw_attn': True,
                       'num_branches': 1,
                       'order': 3,
                       'conv_config': None,
                       'w_linear': False,
                       'sw_linear': True,
                       }
            self.enc = EncAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                   mlp_ratios=[4, 4, 4, 4], depths=depths, setting=setting)
        elif ablation_setting['encoder_module'] == 'lgmm_without_multi_conv':
            setting = {'multi_conv': False,
                       'w_attn': True,
                       'sw_attn': True,
                       'num_branches': 2,
                       'order': 4,
                       'w_linear': False,
                       'sw_linear': True,
                       'conv_config': None,
                       }
            self.enc = EncAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                   mlp_ratios=[4, 4, 4, 4], depths=depths, setting=setting)
        elif ablation_setting['encoder_module'] == 'lgmm_without_w_attn':
            setting = {'multi_conv': True,
                       'w_attn': False,
                       'sw_attn': True,
                       'num_branches': 2,
                       'order': 5,
                       'conv_config': None,
                       'w_linear': False,
                       'sw_linear': True,
                       }
            self.enc = EncAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                   mlp_ratios=[4, 4, 4, 4], depths=depths, setting=setting)
        elif ablation_setting['encoder_module'] == 'lgmm_without_sw_attn':
            setting = {'multi_conv': True,
                       'w_attn': True,
                       'sw_attn': False,
                       'num_branches': 2,
                       'order': 6,
                       'conv_config': None,
                       'w_linear': False,
                       'sw_linear': True,
                       }
            self.enc = EncAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                   mlp_ratios=[4, 4, 4, 4], depths=depths, setting=setting)
        elif ablation_setting['encoder_module'] == 'lgmm_with_ch_split':
            setting = {'multi_conv': True,
                       'w_attn': True,
                       'sw_attn': True,
                       'num_branches': 3,
                       'order': 7,
                       'conv_config': None,
                       'w_linear': False,
                       'sw_linear': True,
                       }
            enc_channels = [66, 96, 129, 258]
            self.enc = EncAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                   mlp_ratios=[4, 4, 4, 4], depths=depths, setting=setting)
        elif ablation_setting['encoder_module'] == 'lgmm_multi_conv_n1':
            setting = {'multi_conv': True,
                       'w_attn': True,
                       'sw_attn': True,
                       'num_branches': 3,
                       'order': 8,
                       'conv_config': {
                           1: {"k": [3, 1], "d": 1},
                           2: {"k": [3, 1], "d": 1},
                           3: {"k": [3, 1], "d": 1},
                           4: {"k": [3, 1], "d": 1}
                       },
                       'w_linear': False,
                       'sw_linear': True,
                       }
            self.enc = EncAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                   mlp_ratios=[4, 4, 4, 4], depths=depths, setting=setting)
        elif ablation_setting['encoder_module'] == 'lgmm_multi_conv_n2':
            setting = {'multi_conv': True,
                       'w_attn': True,
                       'sw_attn': True,
                       'num_branches': 3,
                       'order': 8,
                       'conv_config': {
                           1: {"k": [5, 3, 1], "d": 1},
                           2: {"k": [5, 3, 1], "d": 1},
                           3: {"k": [5, 3, 1], "d": 1},
                           4: {"k": [5, 3, 1], "d": 1}
                       },
                       'w_linear': False,
                       'sw_linear': True,
                       }
            self.enc = EncAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                   mlp_ratios=[4, 4, 4, 4], depths=depths, setting=setting)
        elif ablation_setting['encoder_module'] == 'lgmm_multi_conv_n3':
            setting = {'multi_conv': True,
                       'w_attn': True,
                       'sw_attn': True,
                       'num_branches': 3,
                       'order': 8,
                       'conv_config': {
                           1: {"k": [7, 5, 3, 1], "d": 1},
                           2: {"k": [7, 5, 3, 1], "d": 1},
                           3: {"k": [7, 5, 3, 1], "d": 1},
                           4: {"k": [7, 5, 3, 1], "d": 1}
                       },
                       'w_linear': False,
                       'sw_linear': True,
                       }
            self.enc = EncAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                   mlp_ratios=[4, 4, 4, 4], depths=depths, setting=setting)
        elif ablation_setting['encoder_module'] == 'lgmm_attn_v1':
            setting = {'multi_conv': True,
                       'w_attn': True,
                       'sw_attn': True,
                       'num_branches': 3,
                       'order': 9,
                       'conv_config': None,
                       'w_linear': False,
                       'sw_linear': True, }
            self.enc = EncAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                   mlp_ratios=[4, 4, 4, 4], depths=depths, setting=setting)
        elif ablation_setting['encoder_module'] == 'lgmm_attn_v2':
            setting = {'multi_conv': True,
                       'w_attn': True,
                       'sw_attn': True,
                       'num_branches': 3,
                       'order': 10,
                       'conv_config': None,
                       'w_linear': False,
                       'sw_linear': True,
                       }
            self.enc = EncAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                   mlp_ratios=[4, 4, 4, 4], depths=depths, setting=setting)
        elif ablation_setting['encoder_module'] == 'lgmm_attn_v3':
            setting = {'multi_conv': True,
                       'w_attn': True,
                       'sw_attn': True,
                       'num_branches': 3,
                       'order': 11,
                       'conv_config': None,
                       'w_linear': False,
                       'sw_linear': False,
                       }
            self.enc = EncAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                   mlp_ratios=[4, 4, 4, 4], depths=depths, setting=setting)
        elif ablation_setting['encoder_module'] == 'lgmm_attn_v4':
            setting = {'multi_conv': True,
                       'w_attn': True,
                       'sw_attn': True,
                       'num_branches': 3,
                       'order': 12,
                       'conv_config': None,
                       'w_linear': True,
                       'sw_linear': True,
                       }
            self.enc = EncAblation(patch_size=7, in_chans=in_channels, heads=heads, embed_dims=enc_channels,
                                   mlp_ratios=[4, 4, 4, 4], depths=depths, setting=setting)

        # decoder
        if ablation_setting['decoder_module'] == 'lgmm':
            self.dec = Decoder(in_channels=enc_channels, embedding_dim=dec_embed_dim, output_nc=out_channel)
        elif ablation_setting['decoder_module'] == 'elgc_decoder':
            self.dec = ELGCDecoder(in_channels=enc_channels, embedding_dim=dec_embed_dim, output_nc=out_channel,
                                   use_convT=False)
        elif ablation_setting['decoder_module'] == 'lgmm_without_dynamic_residual':
            setting = {
                'dynamic_residual': False,
                'mask_init': True,
                'angle': 180,
                'mamba_block': None,
                'dirs': 4,
                'use_pyramid': True,
                'use_mgd': True,
            }
            self.dec = DecAblation(in_channels=enc_channels, embedding_dim=dec_embed_dim, output_nc=out_channel,
                                   setting=setting)
        elif ablation_setting['decoder_module'] == 'lgmm_without_mask_init':
            setting = {
                'dynamic_residual': True,
                'mask_init': False,
                'angle': 180,
                'mamba_block': None,
                'dirs': 4,
                'use_pyramid': True,
                'use_mgd': True,
            }
            self.dec = DecAblation(in_channels=enc_channels, embedding_dim=dec_embed_dim, output_nc=out_channel,
                                   setting=setting)
        elif ablation_setting['decoder_module'] == 'lgmm_pmm_n1':
            setting = {
                'dynamic_residual': True,
                'mask_init': True,
                'angle': 180,
                'mamba_block': [1, 1, 1, 1, 1],
                'dirs': 4,
                'use_pyramid': True,
                'use_mgd': True,
            }
            self.dec = DecAblation(in_channels=enc_channels, embedding_dim=dec_embed_dim, output_nc=out_channel,
                                   setting=setting)
        elif ablation_setting['decoder_module'] == 'lgmm_pmm_n2':
            setting = {
                'dynamic_residual': True,
                'mask_init': True,
                'angle': 180,
                'mamba_block': [1, 1, 2, 2, 2],
                'dirs': 4,
                'use_pyramid': True,
                'use_mgd': True,
            }
            self.dec = DecAblation(in_channels=enc_channels, embedding_dim=dec_embed_dim, output_nc=out_channel,
                                   setting=setting)
        elif ablation_setting['decoder_module'] == 'lgmm_pmm_n3':
            setting = {
                'dynamic_residual': True,
                'mask_init': True,
                'angle': 180,
                'mamba_block': [1, 1, 3, 3, 3],
                'dirs': 4,
                'use_pyramid': True,
                'use_mgd': True,
            }
            self.dec = DecAblation(in_channels=enc_channels, embedding_dim=dec_embed_dim, output_nc=out_channel,
                                   setting=setting)
        elif ablation_setting['decoder_module'] == 'lgmm_pmm_rot0':
            setting = {
                'dynamic_residual': True,
                'mask_init': True,
                'angle': 0,
                'mamba_block': [1, 1, 3, 3, 3],
                'dirs': 4,
                'use_pyramid': True,
                'use_mgd': True,
            }
            self.dec = DecAblation(in_channels=enc_channels, embedding_dim=dec_embed_dim, output_nc=out_channel,
                                   setting=setting)
        elif ablation_setting['decoder_module'] == 'lgmm_pmm_rot90':
            setting = {
                'dynamic_residual': True,
                'mask_init': True,
                'angle': 90,
                'mamba_block': [1, 1, 3, 3, 3],
                'dirs': 4,
                'use_pyramid': True,
                'use_mgd': True,
            }
            self.dec = DecAblation(in_channels=enc_channels, embedding_dim=dec_embed_dim, output_nc=out_channel,
                                   setting=setting)
        elif ablation_setting['decoder_module'] == 'lgmm_pmm_rot270':
            setting = {
                'dynamic_residual': True,
                'mask_init': True,
                'angle': 270,
                'mamba_block': [1, 1, 3, 3, 3],
                'dirs': 4,
                'use_pyramid': True,
                'use_mgd': True,
            }
            self.dec = DecAblation(in_channels=enc_channels, embedding_dim=dec_embed_dim, output_nc=out_channel,
                                   setting=setting)
        elif ablation_setting['decoder_module'] == 'lgmm_pmm_2_dirs':
            setting = {
                'dynamic_residual': True,
                'mask_init': True,
                'angle': 270,
                'mamba_block': [1, 1, 3, 3, 3],
                'dirs': 2,
                'use_pyramid': True,
                'use_mgd': True,
            }
            self.dec = DecAblation(in_channels=enc_channels, embedding_dim=dec_embed_dim, output_nc=out_channel,
                                   setting=setting)

        elif ablation_setting['decoder_module'] == 'lgmm_8_dirs':
            setting = {
                'dynamic_residual': True,
                'mask_init': True,
                'angle': 180,
                'mamba_block': [1, 1, 3, 3, 3],
                'dirs': 4,
                'use_pyramid': True,
                'use_mgd': False,
            }
            self.dec = DecAblation(in_channels=enc_channels, embedding_dim=dec_embed_dim, output_nc=out_channel,
                                   setting=setting)
        elif ablation_setting['decoder_module'] == 'lgmm_without_pyramid':
            setting = {
                'dynamic_residual': True,
                'mask_init': True,
                'angle': 180,
                'mamba_block': [1, 1, 3, 3, 3],
                'dirs': 4,
                'use_pyramid': False,
                'use_mgd': True,
            }
            self.dec = DecAblation(in_channels=enc_channels, embedding_dim=dec_embed_dim, output_nc=out_channel,
                                   setting=setting)

    def forward(self, x):
        if isinstance(x, tuple):
            x1, x2 = x
        else:
            x1 = x
            x2 = x
        enc1_out = self.enc(x1)
        enc2_out = self.enc(x2)
        change_map = self.dec(enc1_out, enc2_out)
        if 'elgc_decoder' in self.setting['decoder_module']:
            return change_map
        output, out_128, out_64, out_32, out_16 = change_map
        mask1 = F.interpolate(out_128, scale_factor=2, mode='bilinear', align_corners=False).clamp(min=-20, max=20)
        mask2 = F.interpolate(out_64, scale_factor=4, mode='bilinear', align_corners=False).clamp(min=-20, max=20)
        mask3 = F.interpolate(out_32, scale_factor=8, mode='bilinear', align_corners=False).clamp(min=-20, max=20)
        mask4 = F.interpolate(out_16, scale_factor=16, mode='bilinear', align_corners=False).clamp(min=-20, max=20)
        return [output, mask1, mask2, mask3, mask4]


def get_net(args):
    ablation_setting = {
        'encoder_module': 'lgmm',
        'decoder_module': 'lgmm'
    }
    if args.net == 'lgmm':
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    # encoder 消融
    elif args.net == 'baseline':
        ablation_setting['encoder_module'] = 'msa'
        ablation_setting['decoder_module'] = 'elgc_decoder'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'baseline+dlgpe':
        ablation_setting['encoder_module'] = 'lgmm'
        ablation_setting['decoder_module'] = 'elgc_decoder'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'baseline+pmm':
        ablation_setting['encoder_module'] = 'msa'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_with_resnet18':
        # 使用 resnet18 替换 dlgpe
        ablation_setting['encoder_module'] = 'resnet18'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_with_elgca':
        # 使用 elgca 替换 dlgpe
        ablation_setting['encoder_module'] = 'elgca'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_only_multi_conv':
        ablation_setting['encoder_module'] = 'lgmm_only_multi_conv'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_only_w_attn':
        ablation_setting['encoder_module'] = 'lgmm_only_w_attn'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_only_sw_attn':
        ablation_setting['encoder_module'] = 'lgmm_only_sw_attn'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_without_multi_conv':
        ablation_setting['encoder_module'] = 'lgmm_without_multi_conv'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_without_w_attn':
        ablation_setting['encoder_module'] = 'lgmm_without_w_attn'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_without_sw_attn':
        ablation_setting['encoder_module'] = 'lgmm_without_sw_attn'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_with_ch_split':
        ablation_setting['encoder_module'] = 'lgmm_with_ch_split'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_multi_conv_n1':
        ablation_setting['encoder_module'] = 'lgmm_multi_conv_n1'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_multi_conv_n2':
        ablation_setting['encoder_module'] = 'lgmm_multi_conv_n2'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_multi_conv_n3':
        ablation_setting['encoder_module'] = 'lgmm_multi_conv_n3'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_attn_v1':
        ablation_setting['encoder_module'] = 'lgmm_attn_v1'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_attn_v2':
        ablation_setting['encoder_module'] = 'lgmm_attn_v2'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_attn_v3':
        ablation_setting['encoder_module'] = 'lgmm_attn_v3'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_attn_v4':
        ablation_setting['encoder_module'] = 'lgmm_attn_v4'
        ablation_setting['decoder_module'] = 'lgmm'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)


    # decoder 消融
    elif args.net == 'lgmm_with_elgc_decoder':
        # 使用 ELGCDecoder 替换 pmm
        ablation_setting['encoder_module'] = 'lgmm'
        ablation_setting['decoder_module'] = 'elgc_decoder'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_without_dynamic_residual':
        ablation_setting['encoder_module'] = 'lgmm'
        ablation_setting['decoder_module'] = 'lgmm_without_dynamic_residual'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_without_mask_init':
        ablation_setting['encoder_module'] = 'lgmm'
        ablation_setting['decoder_module'] = 'lgmm_without_mask_init'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_pmm_n1':
        ablation_setting['encoder_module'] = 'lgmm'
        ablation_setting['decoder_module'] = 'lgmm_pmm_n1'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_pmm_n2':
        ablation_setting['encoder_module'] = 'lgmm'
        ablation_setting['decoder_module'] = 'lgmm_pmm_n2'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_pmm_n3':
        ablation_setting['encoder_module'] = 'lgmm'
        ablation_setting['decoder_module'] = 'lgmm_pmm_n3'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_pmm_rot0':
        ablation_setting['encoder_module'] = 'lgmm'
        ablation_setting['decoder_module'] = 'lgmm_pmm_rot0'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_pmm_rot90':
        ablation_setting['encoder_module'] = 'lgmm'
        ablation_setting['decoder_module'] = 'lgmm_pmm_rot90'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_pmm_rot270':
        ablation_setting['encoder_module'] = 'lgmm'
        ablation_setting['decoder_module'] = 'lgmm_pmm_rot270'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_pmm_2_dirs':
        ablation_setting['encoder_module'] = 'lgmm'
        ablation_setting['decoder_module'] = 'lgmm_pmm_2_dirs'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_8_dirs':
        ablation_setting['encoder_module'] = 'lgmm'
        ablation_setting['decoder_module'] = 'lgmm_8_dirs'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    elif args.net == 'lgmm_without_pyramid':
        ablation_setting['encoder_module'] = 'lgmm'
        ablation_setting['decoder_module'] = 'lgmm_without_pyramid'
        net = LGMMNet(3, args.output_nc, ablation_setting=ablation_setting)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net)
    return net.to(device=args.device, dtype=torch.float32)
