import logging
import sys

import torch
# from thop import profile
from ptflops import get_model_complexity_info


class Logger(logging.Logger):
    def __init__(self, name, log_path):
        super().__init__(name, logging.INFO)

        # 配置日志格式
        # formatter = logging.Formatter('%(asctime)s - %(levelname)s - \n%(message)s')
        formatter = logging.Formatter('%(message)s\n')

        # 文件处理器（追加模式）
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)
        self.log_path = log_path

    def write_dict(self, info_dict):
        """写入配置字典到日志"""
        config_msg = "训练配置:\n"
        for k, v in info_dict.items():
            config_msg += f"{k}: {v}\n"
        self.info(config_msg + "\n")

    def write(self, msg):
        if isinstance(msg, list):
            copy = msg
            msg = [[f"{x.item():.2f}" for x in row] for i, row in enumerate(copy)]
        self.info(msg)

    def write_params_flops(self, model):
        """计算并记录模型的参数量和FLOPs"""
        # 1.通过 thop 计算参数
        # intput_tensor = torch.zeros(1, 3, 256, 256).cuda() \
        #                 if torch.cuda.is_available() \
        #                 else torch.zeros(1, 3, 256, 256)
        # flops, params = profile(model, inputs=(intput_tensor,))
        # self.info(f"the parameters of model are {params / 1e6:.4f}M\n")
        # self.info(f"the flops of model are {flops / 1e9:.4f}G\n")

        # 2.通过 ptflops 计算参数
        masc, params = get_model_complexity_info(
            model,
            (3, 256, 256),
            # 强制打印每层统计
            print_per_layer_stat=True,
            # print_per_layer_stat=False,
            as_strings=False,
            # 确保输出详细信息
            verbose=True
            # verbose=False
        )
        self.info(f"the parameters of model are {params / 1e6:.4f}M\n")
        self.info(f"the flops of model are {masc*2 / 1e9:.4f}G\n")
