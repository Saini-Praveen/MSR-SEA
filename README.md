# MSR-SEA
## Multi-Scale Residual Squeeze-and-Excitation Architecture for Underwater Image Enhancement

MSR-SEA is a deep learning architecture for underwater image enhancement. The network combines multi-scale convolutional feature extraction, residual learning, and squeeze-and-excitation (SE) channel attention to process degraded underwater images and reconstruct enhanced RGB images.
<!--
## Architecture

The proposed MSR-SEA model consists of three main stages:
1. **Initial feature extraction:** RGB input is mapped from 3 to 64 channels and then from 64 to 128 channels using 3×3 convolutions.
2. **Multi-Scale Residual (MSR) blocks:** each MSR block extracts features using parallel 3×3 and 5×5 convolutions, concatenates and fuses the resulting features, applies SE channel attention, and performs residual addition.
3. **Image reconstruction:** The resulting features are reduced from 32 to 16 channels and finally reconstructed into a 3-channel RGB output.

The SE block uses global average pooling followed by two fully connected layers with a reduction ratio of 16 and sigmoid channel gating.

## Model Configuration

The released implementation uses:

| Component | Configuration |
|---|---|
| Input | 3-channel RGB image |
| Input resolution | 256 × 256 |
| Initial convolution | 3 → 64, 3 × 3 |
| Second convolution | 64 → 128, 3 × 3 |
| MSR Block 1 | 128 → 64 |
| MSR Block 2 | 64 → 32 |
| Reconstruction convolution | 32 → 16, 3 × 3 |
| Output convolution | 16 → 3, 3 × 3 |
| SE reduction ratio | 16 |
| Activation | ReLU |
-->
## Dataset

The training scripts are configured for the AUIED3K paired underwater image dataset, using corresponding images from `Raw` and `Reference` directories.
The current training configuration uses:
- 3,000 paired images in the complete dataset
- 2,700 training images
- 300 validation images
- 256 × 256 image resolution

  Other datasets we used in this project are UIEB and SYN-TIII.
  Datasets can be downloaded from
  AUIED3K: https://sites.google.com/iiita.ac.in/navjotsingh/research/datasets/auied3k
  UIEB: https://li-chongyi.github.io/proj_benchmark.html
  SYN-TIII: https://li-chongyi.github.io/proj_underwater_image_synthesis.html

  ## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```
## Preparing the Paths

The original research scripts contain machine-specific dataset paths. Before running the code on another system, update the `input_dir` and `target_dir` variables in `train.py` and `test.py` to point to your local AUIED3K directories.

For example:

```python
input_dir = "/path/to/AUIED3K/Raw"
target_dir = "/path/to/AUIED3K/Reference"
```

## Training

Run:

```bash
python train.py
```

The current training configuration is:

```text
Batch size       : 16
Learning rate    : 1e-4
Optimizer        : Adam
Epochs           : 100
Image size       : 256 × 256
Train/Val split  : 2700 / 300
```

The training objective combines SSIM loss and MSE:

```text
L = 0.85 × (1 - SSIM) + 0.15 × MSE
```

Training progress is logged to `msrsea_training_log.csv`, and the checkpoint with the highest validation PSNR is saved as `msrsea_best.pth`.

## Inference

After training, place the trained checkpoint where `test.py` expects it and run:

```bash
python test.py
```

The inference script processes all input images with corresponding reference images, generates enhanced results, and reports the average PSNR, SSIM, MSE, and MAE.

## Image-Quality Evaluation

The repository contains metric implementations for reference-based and no-reference evaluation. The comprehensive evaluation script includes:

- MSE
- PSNR
- SSIM
- UQI
- LPIPS
- UCIQE
- UIQM

Run:

```bash
python metrics.py
```


## Repository Structure

```text
MSR-SEA/
├── README.md
├── requirements.txt
├── .gitignore
├── model.py
├── data.py
├── train.py
├── test.py
├── metrics.py
├── utils.py
├── results/
├── checkpoints/
└── figures/
```

## Pretrained Weights

Pretrained weights can be download from https://drive.google.com/file/d/1epW9UqUe0z_2hA2ue70C-ece8FHn7w1p/view?usp=sharing

## Citation

If you use MSR-SEA in your research, please cite the associated paper:

```bibtex

```

