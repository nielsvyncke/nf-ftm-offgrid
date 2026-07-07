# Neural Field Fourier Token Mixers for Medical Image Segmentation

This repository provides a MONAI bundle for gland segmentation in colon histology images, using data from the [GlaS](https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation) dataset.

Two model variants are supported, selected via the `--model` argument:

- **`swinunetr`**: the standard Swin UNETR [1] architecture.
- **`inr`** (default): a U-Net-style hierarchical encoder--decoder backbone in which each stage's token mixer is a frequency-domain filter parameterized by an implicit neural representation (INR). We refer to this token mixer as **NF-FTM** (Neural Field Fourier Token Mixer): the input is transformed with `torch.fft.rfft2`, multiplied by a learned complex-valued spectral kernel evaluated by a small INR (coordinates -> filter weights), then transformed back with `irfft2`. The INR backbone (`--inr_model`) can be one of:
  - `siren`  - sinusoidal activations (Sitzmann et al., 2020)
  - `wire`   - real Gabor / wavelet activations (Saragadam et al., 2023)
  - `finer`  - variable-periodic activations (Liu et al., 2023)
  - `ffmlp`  - fixed random Fourier features + ReLU MLP (Tancik et al., 2020)

## Overview

This model takes an RGB colon histology image as input and segments gland structures.

### Input Channels: 3
- **0**: Red
- **1**: Green
- **2**: Blue

### Output Channels: 1
- **0**: Gland

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [SLURM Support](#slurm-support)
- [Disclaimer](#disclaimer)
- [References](#references)

## Installation

Run [`setup.sh`](setup.sh) to setup your Python environment and install dependencies. Specify the device (CPU or CUDA) and Python version.

**CPU Setup**:
```bash
bash setup.sh --device cpu --env <env_name> --python_version 3.12
```

**CUDA Setup**:
```bash
bash setup.sh --device cuda --env <env_name> --python_version 3.12
```

## Data

### 1. Download Dataset

The GlaS dataset can be downloaded from [here](https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation). Unpack it so that all `*.bmp` images and `*_anno.bmp` labels are in a single directory (`<data_dir>`).

### 2. Generate Datalist

Use [`scripts/make_datalist.py`](scripts/make_datalist.py) to generate the necessary JSON datalist for training:

```bash
python scripts/make_datalist.py --data_dir /path/to/data
```

This creates the JSON datalist [`configs/datalist.json`](configs/datalist.json) needed for training, with the data partitioned using internal 5-fold, area-stratified cross-validation. A datalist is already provided in this repository for convenience.

## Training

Key default settings (see [`configs/train.yaml`](configs/train.yaml)):

- **Model Input Size**: 256 x 256
- **Optimizer**: AdamW
- **Initial Learning Rate**: 5e-5
- **Loss Function**: DiceCELoss
- **Default Model**: `inr` (SwinUNETR-INR, NF-FTM mixer, SIREN kernel)

### Single-GPU Training
Use [`train.sh`](train.sh) to train on a given data fold using a single GPU:

```bash
bash train.sh --data_dir /path/to/data --fold 0 --batch_size 8
```

Train the plain Swin UNETR baseline instead:

```bash
bash train.sh --data_dir /path/to/data --fold 0 --batch_size 8 --model swinunetr
```

Or train all 5 folds of the default model:

```bash
bash run_folds.sh --data_dir /path/to/data --batch_size 8
```

### Multi-GPU Training
Use [`train_multigpu.sh`](train_multigpu.sh) to train using multiple GPUs (`torchrun` + `DistributedDataParallel`):

```bash
bash train_multigpu.sh --data_dir /path/to/data --fold 0 --batch_size 8
```

## Evaluation

Use [`evaluate.sh`](evaluate.sh) to evaluate a pre-trained model checkpoint on a given fold:

```bash
bash evaluate.sh --data_dir /path/to/data --fold 0 --ckpt_name "model_fold=0.pt"
```

Evaluation runs `configs/evaluate.yaml` on top of `configs/train.yaml`, computing Dice/Hausdorff metrics, saving predictions and per-case metric summaries under the run's `logs/val_fold<k>_<timestamp>/` directory.

## Inference

Use [`inference.sh`](inference.sh) to run ensembled inference over all checkpoints in `models/*.pt` on the test set:

```bash
bash inference.sh --data_dir /path/to/data
```

The predictions will be saved in `<bundle_dir>/outputs` by default.