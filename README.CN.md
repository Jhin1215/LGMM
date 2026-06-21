# **LGMM-Net: A Local-Global Encoder and Mask Mamba Decoder Network for Remote Sensing Change Detection**  
![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green?logo=nvidia)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
> 👥 **作者**：Chen Fang, Shuli Cheng\*, Anyu Du, Chunpeng Wu, Yao Ding  
> 📖 **期刊**：IEEE Transactions on Geoscience and Remote Sensing (TGRS)  
> 🔗 **DOI**：[10.1109/TGRS.2026.3662322](https://doi.org/10.1109/TGRS.2026.3662322)

⭐ **如果您觉得这篇论文对您有帮助，请给一个 star！** 🌟

## 📑 目录

- [🏗️ 网络架构](#️-网络架构)
- [📁 目录结构](#-目录结构)
- [🚀 快速开始](#-快速开始)
  - [1. 克隆仓库](#1-克隆仓库)
  - [2. 环境配置](#2-环境配置)
  - [3. 数据集准备](#3-数据集准备)
  - [4. 推理与训练](#4-推理与训练)
- [🧪 实验结果](#-实验结果)
- [📌 引用](#-引用)
- [📄 开源协议](#-开源协议)

## 🏗️ 网络架构

### 整体框架

<p align="center">
  <img src="./imgs/framework.jpg" alt="LGMM-Net 整体框架" width="900"/>
</p>

### DLGPE 编码器

<p align="center">
  <img src="./imgs/encoder.jpg" alt="DLGPE 编码器" width="900"/>
</p>

### PMM 解码器

<p align="center">
  <img src="./imgs/decoder.jpg" alt="PMM 解码器" width="900"/>
</p>

---

## 📁 目录结构

```text
.
├── models/
│   ├── model.py        # LGMM-Net 整体网络配置（包含消融实验配置）
│   ├── encoder.py      # DLGPE 编码器实现
│   ├── decoder.py      # PMM 解码器实现
│   ├── fusion.py       # 特征融合模块
│   └── model_util.py   # 公共构建模块（ConvLnAct、LayerNorm 等）
├── ablation/
│   ├── ablation_module.py  # 消融实验模块
│   ├── encoder.py          # 消融编码器变体
│   ├── decoder.py          # 消融解码器变体
│   ├── fusion.py           # 消融融合变体
│   ├── resnet.py           # ResNet-18 编码器基线
│   └── rsm_scan.py         # 反向扫描模块
├── tools/
│   ├── timer.py        # 训练计时与统计
│   ├── logger.py       # 日志记录与输出
│   ├── metric_tool.py  # 评价指标计算（F1、IoU 等）
│   └── utils.py        # 损失函数、优化器、学习率调度器等
├── scripts/
│   ├── levir.sh        # LEVIR-CD 训练/推理脚本
│   ├── cdd.sh          # CDD 训练/推理脚本
│   ├── gzcd.sh         # GZCD 训练/推理脚本
│   └── uav.sh          # UAV 训练/推理脚本
├── samples/            # 示例图片（用于快速验证）
├── train.py            # 训练入口
├── inference.py        # 推理入口
├── main.py             # 主入口（训练 + 推理）
├── environment.yml     # Conda 环境配置
└── LICENSE             # Apache License 2.0
```

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/your-name/LGMM_github.git
cd LGMM_github
```

### 2. 环境配置

本项目需要以下核心依赖：

```
python=3.10
pytorch=2.2.0
CUDA=11.8
```

本仓库提供 `environment.yml`，可直接创建并复现论文实验环境：

```bash
# 1) 创建环境
conda env create -f environment.yml
# 2) 激活环境
conda activate lgmm
```

### 3. 数据集准备

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

#### 自动生成 `list/*.txt`（若缺失）

如果你的数据集已经按上述结构放好，但缺少 `list/train.txt` 等文件，可用以下命令自动生成。

> ⚠️ **注意**：需要在数据集根目录（或对应 split 目录）执行。

```bash
# 以 train 为例
mkdir -p train/list
# 有的数据集图片格式可能不是 png，需要自行修改为相应图片格式
basename -a train/A/*.png > train/list/train.txt
```

✅ 我们通常以 `A/` 的文件名作为样本清单来源（因为 `A/B/label` 文件应当同名且一一对应）。

#### 数据集路径设置（必须）

数据集根目录需要在 `tools/dataloader.py` 中的 `get_dataloader` 函数手动修改。请将其中的数据集路径改为你本地的实际路径后，再运行推理/训练脚本。

### 4. 推理与训练

本仓库已提供不同数据集的运行脚本，位于 `scripts/xx.sh`。

由于预训练权重（`checkpoint`）文件夹体积较大，无法直接上传至仓库，现提供以下获取方式（以 **LEVIR-CD** 为例）：

- **百度网盘**：https://pan.baidu.com/s/1PqD-svLTwYy0yWqg-dJH-Q?pwd=gx84  
- **提取码**：`gx84`

如网盘链接失效，请在 issue 中联系我们。

下载后请**直接将整个 `checkpoint/` 文件夹**放到以下目录中（保持文件结构不变）：

```
lgmm_results_LEVIR-CD_epochs_300_lr_1e-3/checkpoint/
```

也就是说，最终你本地应当存在这个文件路径：

```
lgmm_results_LEVIR-CD_epochs_300_lr_1e-3/checkpoint/best_model.pth
```

因此，若使用作者提供的预训练权重进行推理/复现，通常不需要修改任何参数：进入 `scripts` 目录后直接运行对应脚本即可。

```bash
# 以 LEVIR-CD 数据集为例
cd scripts
sh levir.sh
```

#### 从头开始重新训练（可选）

若要重新训练，请修改脚本中的 `save_path`，确保输出目录与作者提供的预训练目录不同。修改 `scripts/xx.sh` 中的 `save_path`，例如：

```bash
save_path=lgmm_${dataset_name}
```

修改完成后，重新执行脚本即可开始训练：

```bash
cd scripts
sh levir.sh
```

#### 超参数设置

模型与训练相关的超参数统一在 `scripts/xxx.sh` 中设置。关键参数如下：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `gpu_ids` | GPU 设备 ID | 各脚本不同 |
| `batch_size` | 批大小 | 16 |
| `lr` | 学习率 | 1e-3 |
| `optimizer` | 优化器 | adamw |
| `lr_mode` | 学习率调度模式 | poly |
| `total_epoch` | 总训练轮数 | 300 |
| `loss_fn` | 损失函数 | ce |
| `input_img_size` | 输入图像尺寸 | 256 |
| `net` | 网络变体 | lgmm |

---

## 🧪 实验结果

### LEVIR-CD、CDD 和 GZ-CD 上的对比

最优结果以**加粗**表示，次优结果以*斜体*表示。所有数值单位为 %。模型按其主要架构特征分组。本方法归类为 Mamba-based，因为 Mamba 解码器构成了核心创新并在建模长距离依赖中起决定性作用。为保证公平比较，分组遵循主导架构范式而非辅助组件的包含情况。"/48" 表示基础通道宽度为 48 的 ConvFormer-CD 变体。
<table>
  <thead>
    <tr>
      <th rowspan="2">Type</th>
      <th rowspan="2">Method</th>
      <th colspan="6">LEVIR-CD</th>
      <th colspan="6">CDD</th>
      <th colspan="6">GZ-CD</th>
    </tr>
    <tr>
      <th>F1</th><th>Rec.</th><th>Pre.</th><th>IoU</th><th>OA</th><th>Kappa</th>
      <th>F1</th><th>Rec.</th><th>Pre.</th><th>IoU</th><th>OA</th><th>Kappa</th>
      <th>F1</th><th>Rec.</th><th>Pre.</th><th>IoU</th><th>OA</th><th>Kappa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><b>CNN-based</b></td>
      <td>FC-EF</td>
      <td>86.31</td><td>84.81</td><td>87.85</td><td>75.91</td><td>98.63</td><td>85.58</td>
      <td>73.10</td><td>60.65</td><td>91.97</td><td>57.60</td><td>94.73</td><td>70.31</td>
      <td>69.89</td><td>57.59</td><td>88.86</td><td>53.71</td><td>95.55</td><td>67.60</td>
    </tr>
    <tr>
      <td>FC-Siam-Diff</td>
      <td>88.42</td><td>86.11</td><td>90.85</td><td>79.24</td><td>98.85</td><td>87.81</td>
      <td>75.50</td><td>63.96</td><td>92.12</td><td>60.64</td><td>95.10</td><td>72.88</td>
      <td>70.34</td><td>57.69</td><td>90.09</td><td>54.25</td><td>95.63</td><td>68.10</td>
    </tr>
    <tr>
      <td>FC-Siam-Conc</td>
      <td>89.29</td><td>88.15</td><td>90.45</td><td>80.64</td><td>98.92</td><td>88.72</td>
      <td>74.63</td><td>64.35</td><td>88.81</td><td>59.52</td><td>94.84</td><td>71.84</td>
      <td>70.45</td><td>58.13</td><td>89.40</td><td>54.38</td><td>95.62</td><td>68.20</td>
    </tr>
    <tr>
      <td rowspan="5"><b>Transformer-based</b></td>
      <td>BIT</td>
      <td>90.25</td><td>88.35</td><td><i>92.23</i></td><td>82.23</td><td>99.03</td><td>89.74</td>
      <td>94.46</td><td>93.78</td><td>95.15</td><td>89.51</td><td>98.70</td><td>93.73</td>
      <td>81.73</td><td>77.08</td><td>86.98</td><td>69.11</td><td>96.91</td><td>80.05</td>
    </tr>
    <tr>
      <td>ChangeFormer</td>
      <td>89.97</td><td>88.26</td><td>91.74</td><td>81.76</td><td>99.00</td><td>89.44</td>
      <td>94.61</td><td>94.51</td><td>94.71</td><td>89.77</td><td>98.73</td><td>93.89</td>
      <td>81.10</td><td>77.08</td><td>85.56</td><td>68.21</td><td>96.78</td><td>79.34</td>
    </tr>
    <tr>
      <td>ELGC-Net</td>
      <td>90.59</td><td>89.10</td><td>92.14</td><td>82.80</td><td>99.06</td><td>90.10</td>
      <td>96.47</td><td><i>96.38</i></td><td>96.56</td><td>93.18</td><td><i>99.17</i></td><td>96.00</td>
      <td>82.23</td><td>76.61</td><td>88.75</td><td>69.83</td><td>97.03</td><td>80.62</td>
    </tr>
    <tr>
      <td>ConvFormer-CD/48</td>
      <td><i>91.08</i></td><td>90.09</td><td>92.10</td><td><i>83.63</i></td><td><i>99.11</i></td><td><i>90.60</i></td>
      <td>96.04</td><td>96.21</td><td>95.87</td><td>92.38</td><td>99.09</td><td>95.51</td>
      <td>85.36</td><td><i>81.80</i></td><td>89.25</td><td>74.46</td><td>97.48</td><td>83.99</td>
    </tr>
    <tr>
      <td>STRobustNet</td>
      <td>90.71</td><td>89.32</td><td>92.15</td><td>83.01</td><td>99.07</td><td>90.22</td>
      <td>93.40</td><td>92.56</td><td>94.25</td><td>87.61</td><td>98.46</td><td>92.52</td>
      <td>81.93</td><td>81.60</td><td>82.26</td><td>69.38</td><td>96.77</td><td>80.15</td>
    </tr>
    <tr>
      <td rowspan="4"><b>Mamba-based</b></td>
      <td>ChangeMamba-B</td>
      <td>90.09</td><td>89.33</td><td>90.86</td><td>81.01</td><td>99.02</td><td>89.49</td>
      <td>95.09</td><td>94.51</td><td>95.68</td><td>90.64</td><td>98.85</td><td>94.44</td>
      <td>83.81</td><td>80.38</td><td>87.56</td><td>72.14</td><td>97.21</td><td>82.29</td>
    </tr>
    <tr>
      <td>RSM-CD</td>
      <td>91.04</td><td><i>90.24</i></td><td>91.85</td><td>83.53</td><td><i>99.11</i></td><td>90.56</td>
      <td><i>96.77</i></td><td>95.86</td><td><b>97.70</b></td><td><i>93.75</i></td><td><i>99.17</i></td><td><i>96.30</i></td>
      <td><i>85.59</i></td><td>80.88</td><td><i>90.87</i></td><td><i>74.81</i></td><td><i>97.56</i></td><td><i>84.26</i></td>
    </tr>
    <tr>
      <td>CDMamba</td>
      <td>90.56</td><td>89.66</td><td>91.47</td><td>82.74</td><td>99.06</td><td>90.06</td>
      <td>95.34</td><td>96.09</td><td>94.59</td><td>91.09</td><td>98.94</td><td>94.72</td>
      <td>84.63</td><td>79.74</td><td>90.17</td><td>73.36</td><td>97.40</td><td>83.22</td>
    </tr>
    <tr>
      <td><b>Ours</b></td>
      <td><b>91.74</b></td><td><b>90.76</b></td><td><b>92.75</b></td><td><b>84.75</b></td><td><b>99.17</b></td><td><b>91.31</b></td>
      <td><b>97.71</b></td><td><b>98.11</b></td><td><i>97.30</i></td><td><b>95.51</b></td><td><b>99.46</b></td><td><b>97.40</b></td>
      <td><b>86.40</b></td><td><b>82.10</b></td><td><b>91.18</b></td><td><b>76.06</b></td><td><b>97.68</b></td><td><b>85.14</b></td>
    </tr>
  </tbody>
</table>



### UAV-CD+ 上的对比

最优结果以**加粗**表示，次优结果以*斜体*表示。所有数值单位为 %。"/48" 表示基础通道宽度为 48 的 ConvFormer-CD 变体。

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Method</th>
      <th>F1</th><th>IoU</th><th>OA</th><th>Kappa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><b>CNN-based</b></td>
      <td>FC-EF</td>
      <td>51.30</td><td>34.50</td><td>92.12</td><td>47.75</td>
    </tr>
    <tr>
      <td>FC-Siam-Diff</td>
      <td>52.65</td><td>35.74</td><td>92.20</td><td>49.04</td>
    </tr>
    <tr>
      <td>FC-Siam-Conc</td>
      <td>51.91</td><td>35.05</td><td>92.50</td><td>48.95</td>
    </tr>
    <tr>
      <td rowspan="5"><b>Transformer-based</b></td>
      <td>BIT</td>
      <td>61.12</td><td>44.01</td><td>93.01</td><td>57.56</td>
    </tr>
    <tr>
      <td>ChangeFormer</td>
      <td>67.60</td><td>51.07</td><td>93.51</td><td>64.06</td>
    </tr>
    <tr>
      <td>ELGC-Net</td>
      <td>67.93</td><td>51.43</td><td>92.33</td><td>64.25</td>
    </tr>
    <tr>
      <td>ConvFormer-CD/48</td>
      <td>69.19</td><td>52.74</td><td>93.28</td><td>65.47</td>
    </tr>
    <tr>
      <td>STRobustNet</td>
      <td>69.06</td><td>52.74</td><td>93.25</td><td>65.56</td>
    </tr>
    <tr>
      <td rowspan="4"><b>Mamba-based</b></td>
      <td>ChangeMamba-B</td>
      <td>68.32</td><td>51.90</td><td>92.74</td><td>64.24</td>
    </tr>
    <tr>
      <td>RSM-CD</td>
      <td>67.31</td><td>50.72</td><td>92.39</td><td>63.61</td>
    </tr>
    <tr>
      <td>CDMamba</td>
      <td><i>69.46</i></td><td><i>53.20</i></td><td><i>93.67</i></td><td><i>66.12</i></td>
    </tr>
    <tr>
      <td><b>Ours</b></td>
      <td><b>70.10</b></td><td><b>53.96</b></td><td><b>93.75</b></td><td><b>66.63</b></td>
    </tr>
  </tbody>
</table>

---

## 📌 引用

如果您觉得这个仓库对您的研究或工作有帮助，请引用我们的论文：

```bibtex
@ARTICLE{lgmm,
  author={Fang, Chen and Cheng, Shuli and Du, Anyu and Wu, Chunpeng and Ding, Yao},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={LGMM-Net: A Local-Global Encoder and Mask Mamba Decoder Network for Remote Sensing Change Detection},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Change detection (CD);Transformer;State Space Model (SSM);Mamba;remote sensing (RS)},
  doi={10.1109/TGRS.2026.3662322}
}
```

---

## 📄 开源协议

本项目基于 [Apache License 2.0](LICENSE) 开源协议。
