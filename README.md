# A Wavelet-Augmented Dual-Branch Position-Embedding Mamba Network for Hyperspectral Image Change Detection
---------------------


  

How to use it?
---------------------

 `python main.py`
# WDP-Mamba


# Abstract
Convolutional neural networks (CNNs) have been widely used in the field of hyperspectral image change detection (HSI-CD) due to their excellent feature extraction capabilities. However, traditional convolution methods usually rely on a limited number of convolution kernels when extracting local features, and the local information only covers a small range of details, making it difficult to flexibly extract local changes at different levels in the image. In addition, CNNs have certain limitations in modeling global dependencies. In recent years, state-space model (SSM)-based methods (such as mamba) have effectively solved the quadratic complexity problem of transformer in capturing long-distance global dependencies. However, due to the processing method of expanding images into sequences, the spatial location structure of the image is destroyed, the spatial correlation between adjacent pixels is weakened, and spatial location information is easily lost, resulting in poor performance
on complex hyperspectral datasets.

To address these issues, we propose a wavelet-augmented dualbranch position-embedding mamba network (WDP-Mamba) for hyperspectral image change detection. This network combines the
complementary advantages of mamba and wavelet convolution (WTconv) in global feature extraction and local edge feature extraction. Specifically, our method includes the following key
innovations: 1) a novel state-space model, the adaptive position residual state space block (APRSSB) is designed, embedding learnable positional encodings within the mamba framework. Through a dual-temporal spatial-spectral scoring mechanism, the model dynamically preserves the topological relationships of pixel positions, enabling global dependency modeling with linear complexity; 2) a multi-level frequency-domain aware (MLFA) module is proposed to fully extract representative local features by decomposing information into frequency components of different scales and analyzing signals locally in multiple time and frequency windows. Extensive experiments on three hyperspectral datasets show that WDP-Mamba outperforms existing methods in both qualitative and quantitative assessments. Compared with other comparative models, WDP-Mamba achieves significant improvements in multiple performance metrics with a lower number of parameters, demonstrating its superior performance in the HSI-CD task.


# Envs
安装causal_conv1d和mamba

pip install -e causal_conv1d>=1.1.0
pip install -e mamba-1p1p1
