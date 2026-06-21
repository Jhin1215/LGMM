# **LGMM-Net: A Local-Global Encoder and Mask Mamba Decoder Network for Remote Sensing Change Detection**  
![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green?logo=nvidia)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
> 👥 **Authors**：Chen Fang, Shuli Cheng\*, Anyu Du\*, Chunpeng Wu, Yao Ding  
> 📖 **Journal**：IEEE Transactions on Geoscience and Remote Sensing (TGRS)  
> 🔗 **DOI**：[10.1109/TGRS.2026.3662322](https://doi.org/10.1109/TGRS.2026.3662322)

⭐ **If you find this paper helpful, please consider giving us a star!** 🌟

## 📑 Table of Contents

- [🏗️ Network Architecture](#️-network-architecture)
- [📁 Directory Structure](#-directory-structure)
- [🚀 Getting Started](#-getting-started)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Environment Setup](#2-environment-setup)
  - [3. Dataset Preparation](#3-dataset-preparation)
  - [4. Inference & Training](#4-inference--training)
- [🧪 Experimental Results](#-experimental-results)
- [📌 Citation](#-citation)
- [📄 License](#-license)

## 🏗️ Network Architecture

### Overall Framework

<p align="center">
  <img src="./imgs/framework.jpg" alt="LGMM-Net Overall Framework" width="900"/>
</p>

### DLGPE Encoder

<p align="center">
  <img src="./imgs/encoder.jpg" alt="DLGPE Encoder" width="900"/>
</p>

### PMM Decoder

<p align="center">
  <img src="./imgs/decoder.jpg" alt="PMM Decoder" width="900"/>
</p>

---

## 📁 Directory Structure

```text
.
├── models/
│   ├── model.py        # LGMM-Net overall network configuration (includes ablation experiment settings)
│   ├── encoder.py      # DLGPE encoder implementation
│   ├── decoder.py      # PMM decoder implementation
│   ├── fusion.py       # Feature fusion module
│   └── model_util.py   # Shared building blocks (ConvLnAct, LayerNorm, etc.)
├── ablation/
│   ├── ablation_module.py  # Ablation experiment modules
│   ├── encoder.py          # Ablation encoder variants
│   ├── decoder.py          # Ablation decoder variants
│   ├── fusion.py           # Ablation fusion variants
│   ├── resnet.py           # ResNet-18 encoder baseline
│   └── rsm_scan.py         # Reverse scanning module
├── tools/
│   ├── timer.py        # Training timing and statistics
│   ├── logger.py       # Logging and output
│   ├── metric_tool.py  # Evaluation metric calculation (F1, IoU, etc.)
│   └── utils.py        # Loss functions, optimizers, LR schedulers, etc.
├── scripts/
│   ├── levir.sh        # LEVIR-CD training/inference script
│   ├── cdd.sh          # CDD training/inference script
│   ├── gzcd.sh         # GZCD training/inference script
│   └── uav.sh          # UAV training/inference script
├── samples/            # Sample images for quick verification
├── train.py            # Training entry point
├── inference.py        # Inference entry point
├── main.py             # Main entry (training + inference)
├── environment.yml     # Conda environment specification
└── LICENSE             # Apache License 2.0
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-name/LGMM_github.git
cd LGMM_github
```

### 2. Environment Setup

The project requires the following core dependencies:

```
python=3.10
pytorch=2.2.0
CUDA=11.8
```

This repository provides `environment.yml` to directly create and reproduce the experimental environment:

```bash
# 1) Create environment
conda env create -f environment.yml
# 2) Activate environment
conda activate lgmm
```

### 3. Dataset Preparation

Using the LEVIR-CD dataset as an example, the project adopts the following directory structure by default:

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

#### Auto-generate `list/*.txt` (if missing)

If your dataset is already organized as above but lacks files like `list/train.txt`, you can use the following command to generate them automatically.

> ⚠️ **Note**: Needs to be executed in the dataset root directory (or corresponding split directory).

```bash
# Take train for example
mkdir -p train/list
# Some datasets may have image formats other than png, modify the format accordingly
basename -a train/A/*.png > train/list/train.txt
```

✅ We typically use filenames in `A/` as the source for the sample list (because files in `A/B/label` should have the same name and correspond one-to-one).

#### Dataset Path Setting (Required)

The dataset root directory needs to be manually modified in the `get_dataloader` function in `tools/dataloader.py`. Please change the dataset path there to your local actual path before running inference/training scripts.

### 4. Inference & Training

This repository provides running scripts for different datasets, located in `scripts/xx.sh`.

Due to the large size of the pretrained weights (the `checkpoint` folder), they cannot be uploaded directly to the repository. Therefore, the following access method is provided (taking **LEVIR-CD** as an example):

- **Baidu Netdisk**: https://pan.baidu.com/s/1PqD-svLTwYy0yWqg-dJH-Q?pwd=gx84  
- **Extraction code**: `gx84`

If the link becomes invalid, please open an issue to contact us.

After downloading, please **directly place the entire `checkpoint/` folder** into the following directory (keeping the folder structure unchanged):

```
lgmm_results_LEVIR-CD_epochs_300_lr_1e-3/checkpoint/
```

That is, the following file path should exist on your local machine:

```
lgmm_results_LEVIR-CD_epochs_300_lr_1e-3/checkpoint/best_model.pth
```

Therefore, if you use the pretrained weights provided by the authors for inference or reproduction, no parameter modification is required. Simply switch to the `scripts` directory and run the corresponding script:

```bash
# Taking the LEVIR-CD dataset as an example
cd scripts
sh levir.sh
```

#### Retraining from Scratch (Optional)

If you want to retrain from scratch, please modify the `save_path` in the script to ensure the output directory is different from the author-provided pretrained directory. Modify `save_path` in `scripts/xx.sh`, for example:

```bash
save_path=lgmm_${dataset_name}
```

After modification, re-execute the script to start training:

```bash
cd scripts
sh levir.sh
```

#### Hyperparameter Settings

Model and training-related hyperparameters are set uniformly in `scripts/xxx.sh`. Key parameters include:

| Parameter | Description | Default |
| --- | --- | --- |
| `gpu_ids` | GPU device ID | Varies by script |
| `batch_size` | Batch size | 16 |
| `lr` | Learning rate | 1e-3 |
| `optimizer` | Optimizer | adamw |
| `lr_mode` | LR scheduler mode | poly |
| `total_epoch` | Total training epochs | 300 |
| `loss_fn` | Loss function | ce |
| `input_img_size` | Input image size | 256 |
| `net` | Network variant | lgmm |

---

## 🧪 Experimental Results

### Comparison on LEVIR-CD, CDD, and GZ-CD

The best results are in **bold** and the next-best are in *italic*. All values are in %. Models are grouped according to their predominant architectural characteristics. Our method is classified as Mamba-based since the Mamba decoder constitutes the core innovation and plays a decisive role in modeling long-range dependencies. For fair comparison, the grouping follows the dominant architectural paradigm rather than the inclusion of auxiliary components. "/48" denotes the ConvFormer-CD variant with a base channel width of 48.

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

### Comparison on UAV-CD+

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

## 📌 Citation

If you find this repository helpful for your research or work, please cite our paper:

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

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
