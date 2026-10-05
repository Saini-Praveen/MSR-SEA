import os
import torch
import numpy as np
from model import MSRSEA
from utils import load_image, save_image, calculate_metrics
from torchvision.transforms import ToTensor, Resize
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

input_dir = '/media/cvblns/NS/Praveen/UW_Datasets/AUIED3K/Raw'
gt_dir = '/media/cvblns/NS/Praveen/UW_Datasets/AUIED3K/Reference'
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

all_files = [f for f in os.listdir(input_dir) if os.path.exists(os.path.join(gt_dir, f))]
print(f"Testing on {len(all_files)} images.")

model = MSRSEA().to(device)
model.load_state_dict(torch.load("msrsea_best.pth", map_location=device))
model.eval()

transform = Resize((256, 256))
to_tensor = ToTensor()

psnr_list, ssim_list, mse_list, mae_list = [], [], [], []
metrics_table = [f"{'Image':<25} {'PSNR':>8} {'SSIM':>8} {'MSE':>12} {'MAE':>12}"]

with torch.no_grad():
    for fname in tqdm(all_files):
        inp_path = os.path.join(input_dir, fname)
        gt_path = os.path.join(gt_dir, fname)

        inp = load_image(inp_path)
        gt = load_image(gt_path)

        if inp is None or gt is None:
            print(f"Skipping {fname} due to load error.")
            continue

        inp_img = Image.fromarray((inp * 255).astype(np.uint8))
        inp_tensor = to_tensor(transform(inp_img)).unsqueeze(0).to(device)

        out_tensor = model(inp_tensor)
        out_np = out_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()

        save_image(out_np, os.path.join(output_dir, fname))

        psnr_val, ssim_val, mse_val, mae_val = calculate_metrics(out_np, gt)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        mse_list.append(mse_val)
        mae_list.append(mae_val)

        metrics_table.append(f"{fname:<25} {psnr_val:8.2f} {ssim_val:8.4f} {mse_val:12.5f} {mae_val:12.5f}")

avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)
avg_mse = np.mean(mse_list)
avg_mae = np.mean(mae_list)

print("\nAverage Metrics:")
print(f"PSNR: {avg_psnr:.2f}")
print(f"SSIM: {avg_ssim:.4f}")
print(f"MSE : {avg_mse:.5f}")
print(f"MAE : {avg_mae:.5f}")

metrics_table.append("-" * 65)
metrics_table.append(f"{'Average':<25} {avg_psnr:8.2f} {avg_ssim:8.4f} {avg_mse:12.5f} {avg_mae:12.5f}")

with open("metrics_eval.txt", "w") as f:
    f.write("\n".join(metrics_table))
