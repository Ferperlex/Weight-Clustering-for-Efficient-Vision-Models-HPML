# Weight-Clustering-for-Efficient-Vision-Models-HPML

## Description
This project explores model compression techniques for a ResNet18 model trained on the CIFAR-10 dataset. The primary focus is on weight clustering methods to reduce the model size while attempting to maintain accuracy.

## Setup
To set up the environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Testing Environment
- **Python version**: 3.12.12
- **GPU type**: NVIDIA A100-SXM4-40GB

## Methods
The following clustering algorithms were applied to the convolutional layer weights:
- **K-Means Clustering**: Tested with k=16, 32, 128.
- **Gaussian Mixture Models (GMM)**: Tested with 16 and 32 components.
- **DBSCAN**: Tested with epsilon values of 0.1 and 0.2.

## Results Summary
- **Baseline**: The uncompressed ResNet18 model achieved ~88.14% accuracy.
- **K-Means**: Provided the best trade-off between compression and accuracy. 
  - k=128 achieved ~34% accuracy with ~4.6x compression.
  - Lower k values improved compression (up to 8x) but significantly reduced accuracy (~25-28%).
- **GMM**: Underperformed compared to K-Means, with accuracies ranging from 15-20%.
- **DBSCAN**: Resulted in model collapse (accuracy ~10-12%), likely due to overly aggressive clustering merging distinct weights.
