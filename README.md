# MTCE-AnomalyDetection
Codes for Hanhui Li, Xinggan Peng, Huiping Zhuang, Zhiping Lin. "Multiple Temporal Context Embedding Networks for Unsupervised Time Series Anomaly Detection", 2022

## Introdution
This reposity demonstrates our proposed Multiple Temporal Context Embedding (MTCE) for unsupervised time series anomaly detection (AD), which can be considered as a flexible plug-in enhance current AD networks with temporal contexts. 

## Requirements
Tested with PyTorch 1.7.1 and a GTX 1080 TI graphics card. 

## Steps to reproduce our results
Here we combine MTCE with Convolutional Autoencoder (Conv-AE) as the example, and evaluate it on the SKAB dataset: 
1. Download the [SKAB](https://github.com/waico/SKAB) dataset.
2. Run ConvAE-PyTorch.py to get the results of our re-implemented PyTorch version of the Conv-AE baseline.
3. Run ConvAE-MTCE-PyTorch.py to get the results of Conv-AE with MTCE (F1 score about 0.83). The implementation details of MTCE can be found in the MTCE class.
