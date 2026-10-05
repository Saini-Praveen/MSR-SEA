import os
import torch
import numpy as np
import csv
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Resize, Compose
from model import MSRSEA
from data import UnderwaterDataset
import torchvision.transforms as transforms
from utils import calculate_metrics
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from pytorch_msssim import ssim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 16
learning_rate = 1e-4
num_epochs = 100
input_dir = '/media/cvblns/NS/Praveen/UW_Datasets/AUIED3K/Raw'
target_dir = '/media/cvblns/NS/Praveen/UW_Datasets/AUIED3K/Reference'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

csv_filename = 'msrsea_training_log.csv'
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'PSNR', 'SSIM', 'MSE', 'MAE'])

def combined_loss(pred, target):
    mse = torch.nn.functional.mse_loss(pred, target)
    ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
    return 0.85 * ssim_loss + 0.15 * mse

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    full_dataset = UnderwaterDataset(input_dir=input_dir, target_dir=target_dir, transform=transform)
    train_size = 2700
    val_size = 300
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MSRSEA().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_psnr = 0.0
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inp, gt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            inp, gt = inp.to(device), gt.to(device)
            optimizer.zero_grad()
            out = model(inp)
            loss = combined_loss(out, gt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Train Loss: {avg_train_loss:.6f}")

        model.eval()
        val_loss = 0.0
        psnr_list, ssim_list, mse_list, mae_list = [], [], [], []

        with torch.no_grad():
            for inp, gt in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                inp, gt = inp.to(device), gt.to(device)
                out = model(inp)
                loss = combined_loss(out, gt)
                val_loss += loss.item()

                for i in range(out.size(0)):
                    pred_np = out[i].cpu().permute(1, 2, 0).numpy()
                    gt_np = gt[i].cpu().permute(1, 2, 0).numpy()
                    p, s, m, a = calculate_metrics(pred_np, gt_np)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    mse_list.append(m)
                    mae_list.append(a)

        avg_val_loss = val_loss / len(val_loader)
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_mse = np.mean(mse_list)
        avg_mae = np.mean(mae_list)

        print(f"Epoch {epoch+1}/{num_epochs} - Avg Val Loss: {avg_val_loss:.6f}")
        print(f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, MSE: {avg_mse:.5f}, MAE: {avg_mae:.5f}")

        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_psnr, avg_ssim, avg_mse, avg_mae])

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "msrsea_best.pth")
            print(f" Model saved at epoch {best_epoch} with Best PSNR {best_psnr:.2f}")

    print(f"\n Best model was saved at epoch {best_epoch} with Best PSNR {best_psnr:.2f}")
