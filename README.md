## 🧾 1. Paper Information

> 📌 **Title**：LGMM-Net: A Local-Global Encoder and Mask Mamba Decoder Network for Remote Sensing Change Detection  
> 👥 **Authors**：Chen Fang, Shuli Cheng\*, Anyu Du, Chunpeng Wu, Yao Ding

---

## 🚀 2. Usage

### 🧩 2.1 Environment

```
python=3.10
pytorch=2.2.0
CUDA=11.8
```

This repository provides environment.yml to directly create and reproduce the experimental environment:

```
# 1) Create environment
conda env create -f environment.yml
# 2) Activate environment
conda activate lgmm
```

### 🗂️ 2.2 Dataset Structure

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

🧾 Auto-generate `list/*.txt` (if missing)
If your dataset is already organized as above but lacks files like `list/train.txt`, you can use the following command
to generate them automatically.

⚠️Note: Needs to be executed in the dataset root directory (or corresponding split directory).

```
# Take train for example
mkdir -p train/list
# Some datasets may have image formats other than png, modify the format accordingly
basename -a train/A/*.png > train/list/train.txt
```

✅ We typically use filenames in A/ as the source for the sample list (because files in `A/B/label` should have the same
name and correspond one-to-one).

### ⚙️ 2.3 Hyperparameter Settings

Model and training-related hyperparameters are set uniformly in `scripts/xxx.sh`.

### 🧠 2.4 Inference and Training

This repository provides running scripts for different datasets, located in `scripts/xx.sh`.  
Due to the large size of the pretrained weights (the `checkpoint` folder), they cannot be uploaded directly to the
repository. Therefore, the following access method is provided (taking **LEVIR-CD** as an example):

Baidu Netdisk download:  
https://pan.baidu.com/s/1PqD-svLTwYy0yWqg-dJH-Q?pwd=gx84  
Extraction code: gx84.  
If the link becomes invalid, you may contact us via email: `2812957539@qq.com`.

After downloading, please **directly place the entire `checkpoint/` folder** into the following directory (keeping the
folder structure unchanged):

`lgmm_results_LEVIR-CD_epochs_300_lr_1e-3/checkpoint/`

That is, the following file path should exist on your local machine:
```lgmm_results_LEVIR-CD_epochs_300_lr_1e-3/checkpoint/best_model.pth```
Therefore, if you use the pretrained weights provided by the authors for inference or reproduction, no parameter
modification is required. Simply switch to the `scripts` directory and run the corresponding script.

```
# Taking the levir-cd dataset as an example
cd scripts
sh levir.sh
```

#### 🔁 2.4.1 Retraining from Scratch (Optional)

If you want to retrain from scratch, please modify the `save_path` in the script to ensure the output directory is
different from the author-provided pretrained directory.
Modify `save_path` in `scripts/xx.sh`, for example:

```save_path=lgmm_${dataset_name}```

After modification, re-execute the script to start training:

```
cd scripts
sh levir.sh
```

#### 📍 2.4.2 Dataset Path Setting (Required)

The dataset root directory needs to be manually modified in the following location:
The `get_dataloader` function in `tools/dataloader.py`.

Please change the dataset path there to your local actual path before running inference/training scripts.

### 🧱 2.4.3 Code Structure Overview

```
.
├── models/
│   ├── model.py        # LGMM-Net overall network configuration (includes related ablation experiment settings)
│   ├── encoder.py      # Implementation code for the LGMM-Net encoder component DLGPE
│   └── decoder.py      # Implementation code for the LGMM-Net decoder component PMM
└── tools/
    ├── timer.py        # Training timing and statistics
    ├── logger.py       # Logging and output
    ├── metric_tool.py  # Evaluation metric calculation
    └── utils.py        # Common utilities like loss functions, optimizers, learning rate schedulers, etc.
```

---

## 📌 3. Citation

If you find this repository helpful for your research or work, please cite our paper:

```
@article{fang202Xlgmm,
  title   = {LGMM-Net: A Local-Global Encoder and Mask Mamba Decoder Network for Remote Sensing Change Detection},
  author  = {Fang, Chen and Cheng, Shuli and Du, Anyu and Wu, Chunpeng and Ding, Yao},
  journal = {TODO},
  year    = {202X}
}
```