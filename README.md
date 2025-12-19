# HPML Project: Weight Clustering for Efficient Vision Models

## Team Information
- **Team Name**: Weight Clustering HPML
- **Members**:
  - Chaeeun Ryu (cr3413)
  - Fernando Macchiavello Cauvi (fm2758)
  - Jinha Park (jp4611)

---

## 1. Problem Statement
This project studies **post-training weight clustering** as a compression technique for a ResNet-18 trained on CIFAR-10.
We replace convolutional-layer weights with a **codebook of centroid values** plus **index assignments** (weight sharing).
We evaluate how clustering affects (1) **test accuracy** and (2) **estimated model size / compression ratio**.

---

## 2. Model Description
- **Backbone**: ResNet-18 variant adapted for CIFAR-10 (3x3 first convolution, stride 1, no max pooling)
- **Framework**: PyTorch
- **Compression target**: all `Conv2d` layers are replaced by a `ClusteredConv2d` layer that reconstructs weights from:
  - trainable centroid parameters
  - frozen per-weight centroid indices

---

## 3. Final Results Summary

The table below matches the values reported in our paper for ResNet-18 on CIFAR-10.

| Setting | Test Accuracy (%) | Model Size (MB) | Compression Ratio (x) |
|---|---:|---:|---:|
| Baseline | 88.14 | 42.66 | 1.00 |
| K-Means (k=16) | 28.06 | 5.42 | 8.00 |
| K-Means (k=32) | 25.62 | 6.75 | 6.40 |
| K-Means (k=128) | 34.18 | 9.41 | 4.57 |
| GMM (16 components) | 20.00 | 5.42 | 8.00 |
| GMM (32 components) | 15.42 | 6.75 | 6.40 |
| DBSCAN (eps=0.1) | 9.80 | 1.42 | 32.00 |

**Observation**: clustering produced large reductions in estimated model size, but accuracy collapsed when clustering was applied uniformly across all convolutional layers and only centroids were fine-tuned.

---

## 4. Reproducibility Instructions

### A. Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Notes on PyTorch wheels:
- Our `requirements.txt` pins CUDA wheels (for our A100 environment). If you are on CPU-only, install an appropriate CPU build of PyTorch/torchvision instead.

### B. WandB Dashboard
We log training curves and experiment summaries here (make sure the project is public):
https://wandb.ai/superbunny38/resnet-compression

### C. Training (Baseline)
Train the baseline ResNet-18 and save weights:
```bash
python train_baseline.py --wandb --wandb-project resnet-compression --save-path baseline_resnet18.pth
```

### D. Clustering + Fine-Tuning (Single Experiment)
Run one clustering configuration (loads the baseline checkpoint, clusters all convolutional layers, then fine-tunes centroids):
```bash
# K-Means with k=16
python run_experiment.py --method kmeans --param 16 --baseline-checkpoint baseline_resnet18.pth --wandb --wandb-project resnet-compression

# GMM with 16 components
python run_experiment.py --method gmm --param 16 --baseline-checkpoint baseline_resnet18.pth --wandb --wandb-project resnet-compression

# DBSCAN with eps=0.1
python run_experiment.py --method dbscan --param 0.1 --baseline-checkpoint baseline_resnet18.pth --wandb --wandb-project resnet-compression
```

### E. Quickstart: Reproduce All Reported Runs
This reproduces the notebook workflow end-to-end: baseline training, then K-Means, GMM, DBSCAN sweeps, plus a summary plot.
```bash
python run_sweep.py --wandb --wandb-project resnet-compression --out-dir outputs
```

Outputs are written to:
- `outputs/results/results.csv`
- `outputs/figures/fig_tradeoff.png` and `outputs/figures/fig_tradeoff.pdf`

### F. Plotting (Standalone)
To generate the trade-off plot from our reported numbers:
```bash
python scripts/plot_tradeoff.py
```

Or generate it from your sweep outputs:
```bash
python scripts/plot_tradeoff.py --csv outputs/results/results.csv --out-dir outputs/figures
```

---

## 5. Repository Outline
- `src/`
  - `models/` ResNet-18 CIFAR-10 implementation
  - `data/` CIFAR-10 dataloaders and preprocessing
  - `training/` training and evaluation loops (with optional WandB logging)
  - `compression/` clustering algorithms, clustered convolution layer, and compression accounting
- `train_baseline.py` baseline training entrypoint
- `run_experiment.py` run one clustering configuration
- `run_sweep.py` reproduce the full sweep and summary plot
- `scripts/plot_tradeoff.py` plotting utility

Contact: fm2758@columbia.edu
