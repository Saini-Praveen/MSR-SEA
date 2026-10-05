import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import torch
import torchvision.transforms as transforms
from PIL import Image
import lpips
import math
from pathlib import Path
from scipy import ndimage

def load_and_resize_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.resize(img, target_size)
    return img

def calculate_mse(img1, img2):
    return mse(img1, img2)

def calculate_psnr(img1, img2):
    return psnr(img1, img2, data_range=255)

def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2, data_range=255)

def calculate_uqi(img1, img2):
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0
    mu1, mu2 = np.mean(img1), np.mean(img2)
    sigma1_sq, sigma2_sq = np.var(img1), np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    numerator = 4 * sigma12 * mu1 * mu2
    denominator = (sigma1_sq + sigma2_sq) * (mu1**2 + mu2**2)
    if denominator < 1e-10:
        return 1.0 if np.allclose(img1, img2) else 0.0
    return numerator / denominator

def calculate_lpips(img1, img2, lpips_model):
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    img1_tensor = transform(Image.fromarray(img1_rgb)).unsqueeze(0)
    img2_tensor = transform(Image.fromarray(img2_rgb)).unsqueeze(0)
    with torch.no_grad():
        score = lpips_model(img1_tensor, img2_tensor)
    return score.item()

def getUCIQE(img):
    img_BGR = img
    img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB).astype(np.float64)
    coe_Metric = [0.4680, 0.2745, 0.2576]
    img_lum = img_LAB[:,:,0]/255.0
    img_a = img_LAB[:,:,1]/255.0
    img_b = img_LAB[:,:,2]/255.0
    chroma = np.sqrt(img_a**2 + img_b**2)
    sigma_c = np.std(chroma)
    img_lum_flat = img_lum.flatten()
    sorted_index = np.argsort(img_lum_flat)
    top_index = sorted_index[int(len(img_lum_flat)*0.99)]
    bottom_index = sorted_index[int(len(img_lum_flat)*0.01)]
    con_lum = img_lum_flat[top_index] - img_lum_flat[bottom_index]
    chroma_flat = chroma.flatten()
    sat = np.divide(chroma_flat, img_lum_flat, out=np.zeros_like(chroma_flat), where=img_lum_flat!=0)
    avg_sat = np.mean(sat)
    uciqe = sigma_c*coe_Metric[0] + con_lum*coe_Metric[1] + avg_sat*coe_Metric[2]
    return uciqe

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    x = sorted(x)
    K = len(x)
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    weight = 1/(K-T_a_L-T_a_R)
    s = int(T_a_L+1)
    e = int(K-T_a_R)
    val = sum(x[s:e])
    val *= weight
    return val

def s_a(x, mu):
    return np.mean((np.array(x)-mu)**2)

def _uicm(x):
    R = x[:,:,0].flatten(); G = x[:,:,1].flatten(); B = x[:,:,2].flatten()
    RG = R-G; YB = (R+G)/2 - B
    mu_a_RG, mu_a_YB = mu_a(RG), mu_a(YB)
    s_a_RG, s_a_YB = s_a(RG, mu_a_RG), s_a(YB, mu_a_YB)
    l = math.sqrt(mu_a_RG**2 + mu_a_YB**2)
    r = math.sqrt(s_a_RG + s_a_YB)
    return (-0.0268*l)+(0.1586*r)

def sobel(x):
    dx = ndimage.sobel(x,0); dy = ndimage.sobel(x,1)
    mag = np.hypot(dx,dy)
    mag *= 255.0/np.max(mag)
    return mag

def eme(x, window_size):
    k1, k2 = int(x.shape[1]/window_size), int(x.shape[0]/window_size)
    w = 2./(k1*k2)
    x = x[:window_size*k2, :window_size*k1]
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_, min_ = np.max(block), np.min(block)
            if min_==0.0 or max_==0.0: continue
            val += math.log(max_/min_)
    return w*val

def _uism(x):
    R,G,B = x[:,:,0], x[:,:,1], x[:,:,2]
    Rs,Gs,Bs = sobel(R), sobel(G), sobel(B)
    r_eme = eme(R*Rs, 10); g_eme = eme(G*Gs,10); b_eme = eme(B*Bs,10)
    return 0.299*r_eme + 0.587*g_eme + 0.144*b_eme

def _uiconm(x, window_size):
    k1, k2 = int(x.shape[1]/window_size), int(x.shape[0]/window_size)
    w = -1./(k1*k2)
    x = x[:window_size*k2, :window_size*k1,:]
    val=0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_, min_ = np.max(block), np.min(block)
            top, bot = max_-min_, max_+min_
            if math.isnan(top) or math.isnan(bot) or bot==0.0 or top==0.0: continue
            val += (top/bot)*math.log(top/bot)
    return w*val

def getUIQM(x):
    x = x.astype(np.float32)
    c1,c2,c3=0.0282,0.2953,3.5753
    uicm = _uicm(x); uism = _uism(x); uiconm = _uiconm(x,10)
    uiqm = c1*uicm + c2*uism + c3*uiconm
    return uiqm


def process_images(image_folder, reference_folder, output_csv="metrics_results.csv"):
    print("Loading LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex')
    results=[]
    image_extensions = {'.jpg','.jpeg','.png','.bmp','.tiff','.tif'}
    image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f.lower())[1] in image_extensions]
    print(f"Found {len(image_files)} images.")
    
    for i, filename in enumerate(image_files,1):
        img_path = os.path.join(image_folder, filename)
        ref_path = os.path.join(reference_folder, filename)
        print(f"Processing: {os.path.basename(img_path)}")
        if not os.path.exists(ref_path):
            print(f"Warning: Reference image not found for {filename}, skipping.")
            continue
        try:
            img = load_and_resize_image(img_path,(256,256))
            ref_img = load_and_resize_image(ref_path,(256,256))
            mse_val = calculate_mse(img, ref_img)
            psnr_val = calculate_psnr(img, ref_img)
            ssim_val = calculate_ssim(img, ref_img)
            uqi_val = calculate_uqi(img, ref_img)
            lpips_val = calculate_lpips(img, ref_img, lpips_model)
            uciqe_val = getUCIQE(img)
            uiqm_val = getUIQM(img)
            results.append({'filename':filename,'MSE':mse_val,'PSNR':psnr_val,'SSIM':ssim_val,
                            'UQI':uqi_val,'LPIPS':lpips_val,'UCIQE':uciqe_val,'UIQM':uiqm_val})
        except Exception as e:
            print(f"Error processing img={img_path}, ref_img={ref_path}: {e}")
            continue
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv,index=False)
        numeric_cols=['MSE','PSNR','SSIM','UQI','LPIPS','UCIQE','UIQM']
        final_metrics = df[numeric_cols].mean()
        final_file = "metrics_final_metrics.txt"
        with open(final_file,'w') as f:
            f.write("FINAL AVERAGE METRICS\n")
            f.write("="*60+"\n")
            for k in numeric_cols:
                f.write(f"{k}: {final_metrics[k]:.6f}\n")
            f.write("="*60+"\n")
        print(f"\nFinal metrics saved to: {final_file}")
        print(df[numeric_cols].describe())
        return df
    else:
        print("No images processed successfully!")
        return None

if __name__ == "__main__":
    image_folder = '/media/cvblns/NS/PraveenUW/MSR-SEA/MSR-SEA-AUIED3K/results'
    reference_folder = '/media/cvblns/NS/Praveen/UW_Datasets/AUIED3K/Reference'
    print("Starting image quality metrics calculation...")
    df = process_images(image_folder, reference_folder)
    if df is not None:
        print("\nProcessing completed successfully!")
