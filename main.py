import sys
import os
from pathlib import Path

import warnings
from argparse import ArgumentParser
from train import fit
sys.path.insert(0, 'tools')
warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue")
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


if __name__ == '__main__':
    parser = ArgumentParser()
    # net setting
    parser.add_argument('--net', default='lgmm_attn_v3', type=str)
    parser.add_argument('--input_img_size', default=256, type=int)
    parser.add_argument('--output_nc', default=2, type=int)
    parser.add_argument('--loss_fn', default='ce', type=str)
    parser.add_argument('--out_sigmoid', default=False, type=bool)
    parser.add_argument('--resume', default=True, type=bool)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--dec_embed_dim', default=256, type=int)
    parser.add_argument('--accum_iters', default=2, type=int, help='梯度累计步数')
    parser.add_argument('--warm_up_epochs', default=0, type=int, help='学习率热身轮数')
    parser.add_argument('--amp_dtype', default='bfloat16', type=str, help='amp 的精度')
    parser.add_argument('--use_amp', default=True, type=bool, help='amp 的精度')

    # lr and optimizer setting
    parser.add_argument('--lr', default=3.1e-4, type=float)
    parser.add_argument('--lr_mode', default='poly', type=str)
    parser.add_argument('--lr_factor', default=1.0, type=float)
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    # dataset setting
    parser.add_argument('--total_epoch', default=1, type=int, help='最大训练轮数')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--dataset_name', default='debug', type=str, help='name of dataset, LEVIR|SYSU|BCDD|CDD')
    parser.add_argument('--root_dir', default='./', type=str)
    parser.add_argument('--use_transform', default=True, type=bool)
    parser.add_argument('--drop_last', default=False, type=bool)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)

    # path setting
    parser.add_argument('--checkpoint_root', default='checkpoint', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)
    parser.add_argument('--save_path', default='./results', type=str)

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    # 创建断点保存目录
    # bebug时，需要讲下述路径覆盖代码注释
    args.save_path = Path("..") / args.save_path
    print(args.save_path)
    args.checkpoint_root = Path(args.save_path) / args.checkpoint_root
    os.makedirs(args.checkpoint_root, exist_ok=True)
    # 创建可视化目录
    args.vis_root = Path(args.save_path) / args.vis_root
    os.makedirs(Path(args.vis_root) / 'pred', exist_ok=True)
    os.makedirs(Path(args.vis_root) / 'label', exist_ok=True)
    os.makedirs(Path(args.vis_root) / 'red', exist_ok=True)
    fit(args)
