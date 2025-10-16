## 🧠 1.LGMM-Net: A Local-Global Encoder and Mask Mamba Decoder Network for Remote Sensing Change Detection


## 📄 2.Paper Information

Authors: Chen Fang, Shuli Cheng*, Anyu Du, Chunpeng Wu, Yao Ding


## 🌟 3.Overview
Remote sensing change detection (CD) is critical
for applications such as urban planning and disaster assess-
ment. It enables accurate identification of changes in ground
objects across bi-temporal images. With the rapid develop-
ment of deep learning, hybrid Convolutional Neural Network
(CNN)–Transformer approaches have significantly advanced this
field. However, existing fusion networks often lack robustness.
They frequently misclassify pseudo-changes as true ones. Even
with hard channel-splitting strategies that jointly model local
and global features, this issue remains unresolved. To address
these challenges, we propose a novel Local-Global Encoder and
Mask Mamba Decoder Network (LGMM-Net). The network
employs multi-kernel convolutions, window attention, and state-
space sequence modeling to achieve robust fusion of local details
and global semantics. First, we introduce a dynamic channel
soft-splitting strategy within the Dynamic Local–Global Parallel
Encoder (DLGPE). This encoder integrates multi-kernel convo-
lutions, fixed-window attention, and shifted-window attention to
capture both local details and global semantics. Second, we design
the Pyramid Mask Mamba (PMM) module with a new scanning
mechanism, termed Mask Guided Directional Selective Scan
(MGD-SS2D). In this module, directional convolutions define
scanning paths, while feature rotations enable four-directional
modeling, thus enhancing boundary representation. Extensive
experiments on three public datasets demonstrate that LGMM-
Net consistently outperforms state-of-the-art (SOTA) methods in
both accuracy and robustness.


## 🚀 4.Highlights

✅ overall architecture:

✅ main contribution 1: DLGPE

✅ main contribution 2: PMM


##  🧩 5.Usage
### 5.1 📁 dataset structure
This project assumes the dataset is organized as follows, using **LEVIR-CD** as an example:
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
│       └── train.txt     # each file name per line, for example 1024_0_0.png
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

### 5.2 trainning

### 5.3 inference