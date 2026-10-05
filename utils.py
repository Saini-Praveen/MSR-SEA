import cv2

import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr

from skimage.metrics import structural_similarity as ssim

from sklearn.metrics import mean_squared_error, mean_absolute_error



def load_image(path, size=(256, 256)):

    img = cv2.imread(path)

    if img is None:

        print(f"Warning: Failed to load {path}")

        return None

    img = cv2.resize(img, size)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img.astype(np.float32)

def save_image(img, path):

    img = (img * 255).clip(0, 255).astype(np.uint8)

    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    

def calculate_metrics(pred, gt):

    pred = np.clip(pred, 0, 1)

    gt = np.clip(gt, 0, 1)



    mse_val = mean_squared_error(gt.flatten(), pred.flatten())

    mae_val = mean_absolute_error(gt.flatten(), pred.flatten())

    psnr_val = psnr(gt, pred, data_range=1.0)

    ssim_val = ssim(gt, pred, data_range=1.0, channel_axis=-1)

    return psnr_val, ssim_val, mse_val, mae_val

