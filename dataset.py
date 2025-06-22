import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import astropy.io.fits as fits
import numpy as np
import torch
from config import *
import matplotlib.pyplot as plt
import time

def save_loss_to_txt(
        train_loss, val_loss, 
        train_mse_loss, val_mse_loss,
        train_bpp_loss, val_bpp_loss,
        train_loss_log_path=train_loss_log_path, 
        val_loss_log_path=val_loss_log_path,
        train_loss_mse_log_path=train_loss_mse_log_path,
        val_loss_mse_log_path=val_loss_mse_log_path,
        train_loss_bpp_log_path=train_loss_bpp_log_path,
        val_loss_bpp_log_path=val_loss_bpp_log_path
        ):
    with open(train_loss_log_path, 'a') as f:
        f.write(f"{train_loss}\n")
    with open(val_loss_log_path, 'a') as f:
        f.write(f"{val_loss}\n")
    with open(train_loss_mse_log_path, 'a') as f:
        f.write(f"{train_mse_loss}\n")
    with open(val_loss_mse_log_path, 'a') as f:
        f.write(f"{val_mse_loss}\n")
    with open(train_loss_bpp_log_path, 'a') as f:
        f.write(f"{train_bpp_loss}\n")
    with open(val_loss_bpp_log_path, 'a') as f:
        f.write(f"{val_bpp_loss}\n")    


def read_loss_file(filepath):
    losses = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                losses.append(float(line.strip()))
    return losses


def plot_dual_axis(train_losses, val_losses, selected_epoch = 389, mode = "all_loss"):
    epochs = list(range(1,len(val_losses)+1))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    if mode == "all_loss":
        val_label_name = "Validation Loss"
        train_label_name = "Train Loss"
        title_name = "Training vs Validation Loss (Dual Y-Axis)"
        save_path = 'loss_plot_all_loss.png'
    elif mode == "mse_loss":
        val_label_name = "Validation MSE Loss"
        train_label_name = "Train MSE Loss"
        title_name = "Training vs Validation MSE Loss (Dual Y-Axis)"
        save_path = 'loss_plot_mse_loss.png'
    elif mode == "bpp_loss":
        val_label_name = "Validation BPP Loss"
        train_label_name = "Train BPP Loss"
        title_name = "Training vs Validation BPP Loss (Dual Y-Axis)"
        save_path = 'loss_plot_bpp_loss.png'
    else:
        raise ValueError("Invalid mode. Choose from 'all_loss', 'mse_loss', or 'bpp_loss'.")
    
    # 파일 이름에 타임스탬프 추가
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(save_path)
    save_path = f"{name}_{timestamp}{ext}"
    directory = "graph/"

    # set outlier
    tmean = np.mean(train_losses)
    tstd = np.std(train_losses)
    vmean = np.mean(val_losses)
    vstd = np.std(val_losses)

    # 왼쪽 y축 - train Loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(train_label_name, color='tab:blue')
    ax1.plot(epochs, train_losses, label=train_label_name, color='tab:blue', marker='o', alpha = 0.7)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    # ax1.set_ylim(tmean - 1*tstd, tmean + 1 * tstd)  # y축 범위 설정 (outlier 제거)

    # 오른쪽 y축 - validation Loss
    ax2 = ax1.twinx()  # ax1과 x축 공유, y축만 따로
    ax2.set_ylabel(val_label_name, color='tab:red')
    ax2.plot(epochs, val_losses, label=val_label_name, color='tab:red', marker='o', alpha = 0.7)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.scatter(epochs[selected_epoch-1], val_losses[selected_epoch-1], marker='*', s=200, label='model_selected_1', color='orange', zorder=3) # 모델 선택 지점 표시
    # ax2.set_ylim(vmean - 1*vstd, vmean + 1 * vstd)  # y축 범위 설정 (outlier 제거)

    # 제목 및 저장
    ax1.grid()
    plt.title(title_name)
    fig.tight_layout()  # 여백 자동 조정
    plt.savefig(directory + save_path)
    plt.close()
    print(f"graph saved: {directory+save_path}")

class ImageDataset(Dataset):
    def __init__(self, img_dir, data_name_lst, transform=None):
        self.img_dir = img_dir
        self.img_labels = data_name_lst 
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        if img_path.endswith("npz"):
            image = np.load(img_path)['x']
        else:
            image = fits.open(img_path)[0].data

        if self.transform:
            image = self.transform(image)
        return image

def load_file_list(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def get_datasets(txt_path = data1024_filename, dir = PATH, train_ratio=0.7, val_ratio=0.2, transform=None):
    # random split
    # split된 이름리스트로 ImageDataset return

    full_list = load_file_list(txt_path)
    total = len(full_list)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val
    generator1 = torch.Generator().manual_seed(42)

    train_name_lst, val_name_lst, test_name_lst = random_split(full_list, [n_train, n_val, n_test], generator=generator1)
    
    train_set = ImageDataset(dir, train_name_lst, transform=transform)
    val_set = ImageDataset(dir, val_name_lst, transform=transform)
    test_set = ImageDataset(dir, test_name_lst, transform=transform)
    
    return train_set, val_set, test_set
