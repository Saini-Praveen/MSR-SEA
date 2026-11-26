# MSR-SEA
## Multi-Scale Residual Squeeze-and-Excitation Architecture for Underwater Image Enhancement

MSR-SEA is a deep learning architecture for underwater image enhancement. The network combines multi-scale convolutional feature extraction, residual learning, and squeeze-and-excitation (SE) channel attention to process degraded underwater images and reconstruct enhanced RGB images. The details of the architecture are shown in the paper and also found in `model.py`.
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

The training and evaluation scripts are configured for the UIEB paired underwater image enhancement dataset, containing corresponding underwater images in `Raw` and `Reference` directories.

The dataset is organized into separate training, validation, and testing subsets:

```text
UIEB/
├── train/
│   ├── Raw/
│   └── Reference/
├── val/
│   ├── Raw/
│   └── Reference/
└── test/
    ├── Raw/
    └── Reference/
```

The dataset consists of 890 paired images, divided as follows:

- **Training set:** 623 image pairs (70%)
- **Validation set:** 89 image pairs (10%)
- **Testing set:** 178 image pairs (20%)
- **Image resolution:** 256 × 256 pixels

The dataset is divided into mutually exclusive training, validation, and testing subsets. The corresponding image filenames for each subset are provided in `dataset_split_file_lists.txt` to ensure reproducibility of the dataset split.

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

Before running the code, update the dataset root path in `train.py` and `test.py` to point to your local dataset root directory. Ensure that the subdirectories train, val, and test follow the directory structure mentioned above.


For example, configure the paths in `train.py` and  `test.py` as follows:

```python
dataset_root = "/path/to/dataset"

```

Replace `/path/to/dataset` with the actual location of your dataset.

Also, configure the path of enhanced test images and the corresponding reference images in `calculate_metrics.py`


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
```

The training objective combines SSIM loss and MSE:

```text
L = 0.75 × (1 - SSIM) + 0.25 × MSE
```

Training progress is logged to `msrsea_training_log.csv`, and the checkpoint with the highest validation PSNR is saved as `msrsea_best.pth`.

## Inference

After training, place the trained checkpoint where `test.py` expects it and run:

```bash
python test.py
```

The inference script processes test set images with corresponding reference images, generates enhanced results, and reports the average PSNR, SSIM, MSE, and MAE, along with inference time.

## Image-Quality Evaluation

The repository contains metric implementations for reference-based and non-reference evaluation. The comprehensive evaluation script includes:

- MSE
- PSNR
- SSIM
- UQI (Grayscale and Color both)
- LPIPS
- UCIQE
- UIQM

Run:

```bash
python calculate_metrics.py
```
MSR-SEA reported all reference and non-reference metric scores calculated using `calculate_metrics.py`

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

Pretrained weights can be downloaded from [Google Drive](https://drive.google.com/file/d/1JfR9puX1nU2qvdRfbGg1uEOHVd1GlHK1/view?usp=sharing).

## Additional Structural Analysis

`sift.py` and `sobel.py` are code for the scale-invariant feature transform and Sobel edge detection, which need to be executed separately on the dataset directories (test - Raw and Reference), along with directories containing images of the MSR-SEA-enhanced and competing methods. Configure the paths to the directories in the corresponding code for a particular dataset. 

## Citation

If you use MSR-SEA in your research, please cite the associated paper:

```bibtex
loading
```

