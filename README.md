# MTCE-AnomalyDetection
Codes for Hanhui Li, Xinggan Peng, Huiping Zhuang, Zhiping Lin. "Multiple Temporal Context Embedding Networks for Unsupervised Time Series Anomaly Detection", ICASSP 2022

## Introduction
This reposity demonstrates our proposed Multiple Temporal Context Embedding (MTCE) method for unsupervised time series anomaly detection (AD), which can be considered as a flexible plug-in to enhance current AD networks in exploiting temporal contexts. 

## Requirements
Tested with PyTorch 1.7.1 and a GTX 1080 TI graphics card. 

## Explanation
The proposed MTCE module aims at combing multiple temporal contexts to better tackle anomaly detection. It can be combined with different state-of-the-art AD networks as follows: Left: [Conv-AE](https://github.com/waico/SKAB); Right: [MTAD-GAT](https://github.com/ML4ITS/mtad-gat-pytorch).
![usage](/imgs/usage.png "Combining the MTCE module with different AD networks.")

Here we explain the procedure of MTCE with Conv-AE as the baseline. Following the Conv-AE example on SKAB, we construct a Conv-AE with an encoder of two convolutional layers (input dim -> 32 -> 16), and a decoder of three convolutional layers (16 -> 16 -> 32 -> input dim). Details of this Conv-AE can be found in "ConvAE-PyTorch.py", L108-L126.

The implementation of MTCE can be found in "ConvAE-MTCE-PyTorch.py". Our MTCE uses all hidden features in the encoder (i.e., 32D and 16D) to generate the embedded features (16D), and send them to the decoder as the original features, as follows:
![module](/imgs/module.png "Diagram of the MTCE module with Conv-AE.")

To this end, we  

  a) first project all hidden features into the same feature space via convolutional layers.  
  
  b) Reconstruct the output hidden features of the encoder (i.e., 16D) via the linear weighted combination of base features, where the weights are dot-product similarity, and base   features are features that have top-k similarities (L144-L148 in "ConvAE-MTCE-PyTorch.py").  
  
  c) The reconstructed features are transformed via a convolutional layer and added back to the output hidden features, as the embedded features.  
  
  d) The original features and the embedded features are used in the loss functions (L215) and the thresholding for detecting anomalies (L227-L230).


## Steps to reproduce our results
Here we combine MTCE with Convolutional Autoencoder (Conv-AE) as the example, and evaluate it on the SKAB dataset: 
1. Download the [SKAB](https://github.com/waico/SKAB) dataset.
2. Run ConvAE-PyTorch.py to get the results of our re-implemented PyTorch version of the Conv-AE baseline.
3. Run ConvAE-MTCE-PyTorch.py to get the results of Conv-AE with MTCE (F1 score about 0.83). The implementation details of MTCE can be found in the MTCE class.
