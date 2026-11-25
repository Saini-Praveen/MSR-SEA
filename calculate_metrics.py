import os
import csv
import math
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import lpips
from pathlib import Path

from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
    mean_squared_error as mse,
)
from scipy import ndimage


IMAGE_SIZE = (256, 256)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

UCIQE_C1, UCIQE_C2, UCIQE_C3 = 0.4680, 0.2745, 0.2576

UIQM_C1, UIQM_C2, UIQM_C3 = 0.0282, 0.2953, 3.5753

UICM_ALPHA_L = 0.1
UICM_ALPHA_R = 0.1
UIQM_BLOCK_SIZE = 10
UISM_LAMBDA_RGB = (0.299, 0.587, 0.114)

def load_and_resize_image(image_path, target_size=IMAGE_SIZE):
    """Load an image as uint8 BGR and resize it to (width, height)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


def calculate_mse(img1, img2):
    return float(mse(img1, img2))


def calculate_psnr(img1, img2):
    return float(psnr(img1, img2, data_range=255))


def calculate_ssim(img1, img2):
    return float(ssim(img1, img2, channel_axis=2, data_range=255))


def _uqi_single_channel(img1, img2):
    x = np.asarray(img1, dtype=np.float64)
    y = np.asarray(img2, dtype=np.float64)

    mu_x, mu_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)
    cov_xy = np.mean((x - mu_x) * (y - mu_y))

    denominator = (var_x + var_y) * (mu_x**2 + mu_y**2)
    numerator = 4.0 * cov_xy * mu_x * mu_y

    if denominator < 1e-12:
        return 1.0 if np.array_equal(x, y) else 0.0
    return float(numerator / denominator)


def calculate_uqi_grayscale(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
    return _uqi_single_channel(gray1, gray2)


def calculate_uqi_color(img1, img2):
    if img1.shape != img2.shape or img1.ndim != 3 or img1.shape[2] != 3:
        raise ValueError("Colour UQI expects matching three-channel images.")
    return float(np.mean([
        _uqi_single_channel(img1[:, :, c], img2[:, :, c])
        for c in range(3)
    ]))


def calculate_lpips(img1_bgr, img2_bgr, lpips_model, device):
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    t1 = transform(Image.fromarray(img1_rgb)).unsqueeze(0).to(device)
    t2 = transform(Image.fromarray(img2_rgb)).unsqueeze(0).to(device)

    with torch.inference_mode():
        score = lpips_model(t1, t2)
    return float(score.item())


def getUCIQE(img_bgr):
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("UCIQE expects a valid three-channel BGR image.")

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    L = lab[..., 0] / 255.0
    a = (lab[..., 1] - 128.0) / 255.0
    b = (lab[..., 2] - 128.0) / 255.0

    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = chroma.std()

    l_sorted = np.sort(L.ravel())
    n = max(1, int(round(l_sorted.size * 0.01)))
    con_l = l_sorted[-n:].mean() - l_sorted[:n].mean()

    nz = L > 0
    mu_s = (chroma[nz] / L[nz]).mean() if nz.any() else 0.0

    return float(UCIQE_C1 * sigma_c + UCIQE_C2 * con_l + UCIQE_C3 * mu_s)


def _uicm(img_rgb):
    img = img_rgb.astype(np.float64)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    rg = (R - G).ravel()
    yb = ((R + G) / 2.0 - B).ravel()

    def stats(x):
        x = np.sort(x)
        n = x.size
        lo = int(math.ceil(UICM_ALPHA_L * n))
        hi = int(math.floor(UICM_ALPHA_R * n))
        mu = x[lo:n - hi].mean() if (n - hi) > lo else x.mean()
        var = ((x - mu) ** 2).mean()
        return mu, var

    mu_rg, var_rg = stats(rg)
    mu_yb, var_yb = stats(yb)
    mu = math.sqrt(mu_rg ** 2 + mu_yb ** 2)
    sigma = math.sqrt(var_rg + var_yb)
    return -0.0268 * mu + 0.1586 * sigma


def _eme(x, bs=UIQM_BLOCK_SIZE):
    h, w = x.shape
    kh, kw = h // bs, w // bs
    if kh == 0 or kw == 0:
        return 0.0
    blocks = x[:kh * bs, :kw * bs].reshape(kh, bs, kw, bs)
    mx = blocks.max(axis=(1, 3))
    mn = blocks.min(axis=(1, 3))
    valid = (mx > 0) & (mn > 0)
    val = np.zeros_like(mx)
    val[valid] = np.log(mx[valid] / mn[valid])
    return 2.0 * val.mean()


def _sobel_norm(ch):
    mag = np.hypot(ndimage.sobel(ch, 0), ndimage.sobel(ch, 1))
    m = mag.max()
    return mag * (255.0 / m) if m > 0 else mag


def _uism(img_rgb):
    img = img_rgb.astype(np.float64)
    total = 0.0
    for c, lam in enumerate(UISM_LAMBDA_RGB):
        ch = img[..., c]
        total += lam * _eme(_sobel_norm(ch) * ch, UIQM_BLOCK_SIZE)
    return total


def _uiconm(img_rgb, bs=UIQM_BLOCK_SIZE):
    x = img_rgb.astype(np.float64)
    h, w = x.shape[:2]
    kh, kw = h // bs, w // bs
    if kh == 0 or kw == 0:
        return 0.0
    blocks = x[:kh * bs, :kw * bs].reshape(kh, bs, kw, bs, -1)
    mx = blocks.max(axis=(1, 3, 4))
    mn = blocks.min(axis=(1, 3, 4))
    top, bot = mx - mn, mx + mn
    valid = (top > 0) & (bot > 0)
    val = np.zeros_like(mx)
    r = top[valid] / bot[valid]
    val[valid] = r * np.log(r)
    return float(-val.mean())


def getUIQM(img_bgr):
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("UIQM expects a valid three-channel BGR image.")

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)

    uicm = _uicm(rgb)
    uism = _uism(rgb)
    uiconm = _uiconm(rgb)

    return float(UIQM_C1 * uicm + UIQM_C2 * uism + UIQM_C3 * uiconm)


def process_images(image_folder, reference_folder, output_csv="metrics_results.csv"):
    image_folder = Path(image_folder)
    reference_folder = Path(reference_folder)

    if not image_folder.is_dir():
        raise FileNotFoundError(f"Enhanced-image directory not found: {image_folder}")
    if not reference_folder.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {reference_folder}")

    print("Loading LPIPS model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    image_files = sorted(
        p for p in image_folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"Found {len(image_files)} enhanced images.")

    results = []
    missing_references = []
    failed_images = []

    for i, img_path in enumerate(image_files, 1):
        ref_path = reference_folder / img_path.name
        print(f"Processing [{i}/{len(image_files)}]: {img_path.name}")

        if not ref_path.is_file():
            print(f"Warning: reference not found: {ref_path}")
            missing_references.append(img_path.name)
            continue

        try:
            img = load_and_resize_image(img_path)
            ref_img = load_and_resize_image(ref_path)

            mse_val = calculate_mse(img, ref_img)
            psnr_val = calculate_psnr(img, ref_img)
            ssim_val = calculate_ssim(img, ref_img)
            uqi_gray_val = calculate_uqi_grayscale(img, ref_img)
            uqi_color_val = calculate_uqi_color(img, ref_img)
            lpips_val = calculate_lpips(img, ref_img, lpips_model, device)

            uciqe_val = getUCIQE(img)
            uiqm_val = getUIQM(img)

            results.append({
                "filename": img_path.name,
                "MSE": mse_val,
                "PSNR": psnr_val,
                "SSIM": ssim_val,
                "UQI_Gray": uqi_gray_val,
                "UQI_Color": uqi_color_val,
                "LPIPS": lpips_val,
                "UCIQE": uciqe_val,
                "UIQM": uiqm_val,
            })

        except Exception as exc:
            print(f"Error processing {img_path.name}: {exc}")
            failed_images.append(img_path.name)

    if not results:
        print("No images processed successfully.")
        return None

    df = pd.DataFrame(results)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    numeric_cols = [
        "MSE", "PSNR", "SSIM", "UQI_Gray", "UQI_Color",
        "LPIPS", "UCIQE", "UIQM",
    ]
    averages = df[numeric_cols].mean()

    final_file = output_csv.parent / "metrics_final_metrics.txt"
    with open(final_file, "w", encoding="utf-8") as f:
        f.write("FINAL AVERAGE METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Successfully evaluated image pairs: {len(df)}\n")
        f.write(f"Input images found: {len(image_files)}\n")
        f.write(f"Missing references: {len(missing_references)}\n")
        f.write(f"Failed images: {len(failed_images)}\n")
        f.write("-" * 60 + "\n")
        for key in numeric_cols:
            f.write(f"{key}: {averages[key]:.6f}\n")
        f.write("=" * 60 + "\n")

    print(f"\nPer-image metrics saved to: {output_csv}")
    print(f"Final average metrics saved to: {final_file}")
    print(df[numeric_cols].describe())

    if missing_references:
        print(f"\nMissing references ({len(missing_references)}):")
        for name in missing_references:
            print(f"  {name}")

    if failed_images:
        print(f"\nFailed images ({len(failed_images)}):")
        for name in failed_images:
            print(f"  {name}")

    return df


if __name__ == "__main__":
    image_folder = "results"
    reference_folder = ("./UIEB/test/Reference")

    print("Starting image quality metrics calculation...")
    df = process_images(
        image_folder=image_folder,
        reference_folder=reference_folder,
        output_csv="metrics_results.csv",
    )

    if df is not None:
        print("\nProcessing completed successfully!")
