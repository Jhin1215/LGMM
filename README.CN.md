## 🧾 1. 论文信息

> 📌 **Title**：LGMM-Net: A Local-Global Encoder and Mask Mamba Decoder Network for Remote Sensing Change Detection  
> 👥 **Authors**：Chen Fang, Shuli Cheng\*, Anyu Du, Chunpeng Wu, Yao Ding

---

## 🚀 2. 使用说明

### 🧩 2.1 运行环境

```
python=3.10
pytorch=2.2.0
CUDA=11.8
```

本仓库提供 environment.yml，可直接创建并复现论文实验环境：

```
# 1) 创建环境
conda env create -f environment.yml
# 2) 激活环境
conda activate lgmm
```

### 🗂️ 2.2 数据集结构

以 LEVIR-CD 数据集为例，项目默认采用如下目录结构：

```
LEVIR-CD/
├── train/
│   ├── A/
│   │   ├── 1024_0_0.png
│   │   └── ...
│   ├── B/
│   │   ├── 1024_0_0.png
│   │   └── ...
│   ├── label/
│   │   ├── 1024_0_0.png
│   │   └── ...
│   └── list/
│       └── train.txt
├── val/
│   ├── A/
│   ├── B/
│   ├── label/
│   └── list/
│       └── val.txt
└── test/
    ├── A/
    ├── B/
    ├── label/
    └── list/
        └── test.txt
```

🧾 自动生成 `list/*.txt`（若缺失）
如果你的数据集已经按上述结构放好，但缺少 `list/train.txt` 等文件，可用以下命令自动生成。

⚠️注 ：需要在数据集根目录（或对应 split 目录）执行。

```
# 以 train 为例
mkdir -p train/list
# 有的数据集图片格式可能不是 png，需要自行修改为相应图片格式
basename -a train/A/*.png > train/list/train.txt
```

✅ 说明：我们通常以 A/ 的文件名作为样本清单来源（因为 `A/B/label` 文件应当同名且一一对应）。

### ⚙️ 2.3 超参数设置

模型与训练相关的超参数统一在 `scripts/xxx.sh` 中设置。

### 🧠 2.4 推理与训练

本仓库已提供不同数据集的运行脚本，位于 `scripts/xx.sh`。
由于预训练权重（`checkpoint`）文件夹体积较大，无法直接上传至仓库，现提供以下获取方式（以 **LEVIR-CD** 为例）：

百度网盘下载：https://pan.baidu.com/s/1PqD-svLTwYy0yWqg-dJH-Q?pwd=gx84
提取码: gx84。如网盘链接失效可通过邮箱联系获取：`2812957539@qq.com`

下载后请**直接将整个 `checkpoint/` 文件夹**放到以下目录中（保持文件结构不变）：

`lgmm_results_LEVIR-CD_epochs_300_lr_1e-3/checkpoint/`

也就是说，最终你本地应当存在这个文件路径：

```lgmm_results_LEVIR-CD_epochs_300_lr_1e-3/checkpoint/best_model.pth```

因此，若使用作者提供的预训练权重进行推理/复现，通常不需要修改任何参数：进入 `scripts` 目录后直接运行对应脚本即可。

```
# 以 levir-cd 数据集为例
cd scripts
sh levir.sh
```

#### 🔁 2.4.1 从头开始重新训练（可选）

若要重新训练，请修改脚本中的 `save_path`，确保输出目录与作者提供的预训练目录不同。
修改 `scripts/xx.sh` 中的 `save_path`，例如：

```save_path=lgmm_${dataset_name}```

修改完成后，重新执行脚本即可开始训练：

```
cd scripts
sh levir.sh
```

#### 📍 2.4.2 数据集路径设置（必须）

数据集根目录需要在下列位置手动修改：
`tools/dataloader.py` 中的 `get_dataloader` 函数。

请将其中的数据集路径改为你本地的实际路径后，再运行推理/训练脚本。

### 🧱 2.4.3 代码结构说明

```
.
├── models/
│   ├── model.py        # LGMM-Net 整体网络配置（包含相关消融实验配置）
│   ├── encoder.py      # LGMM-Net 编码器组件 DLGPE 的实现代码
│   └── decoder.py      # LGMM-Net 解码器组件 PMM 的实现代码
└── tools/
    ├── timer.py        # 训练计时与统计
    ├── logger.py       # 日志记录与输出
    ├── metric_tool.py  # 评价指标计算
    └── utils.py        # 损失函数、优化器、学习率调度器等通用工具
```

---

## 📌 3. 引用

如果你觉得这个仓库对你的研究或工作有帮助，请引用我们的论文：

```
@article{fang202Xlgmm,
  title   = {LGMM-Net: A Local-Global Encoder and Mask Mamba Decoder Network for Remote Sensing Change Detection},
  author  = {Fang, Chen and Cheng, Shuli and Du, Anyu and Wu, Chunpeng and Ding, Yao},
  journal = {TODO},
  year    = {202X}
}
```