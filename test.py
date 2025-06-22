# test pretrained model, finetuned model

from dataset import get_datasets
from transforms import get_test_transform, get_ori_transform
from compressai.zoo import cheng2020_attn
from torch.utils.data import DataLoader
from metric import calculate_ms_ssim, calculate_psnr
import torch
import numpy as np
from config import PATH
import math
# from skimage.transform import rescale
# from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib.pyplot as plt

def unpreprocess(maxv_log, minv_log, res_img):
    res_img = res_img[0][0] # (B,3,H,W) -> (H,W)
    res_img = res_img * (maxv_log - minv_log) + minv_log
    res_img = torch.expm1(res_img)
    return res_img

def load_file_list(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def calculate_bits(likelihood):
    return torch.sum(-torch.log2(likelihood)).item()

data1024_filename = 'data1024lst.txt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fine tuned model 갖고오기
model_finetuned = cheng2020_attn(pretrained=False, quality=6).to(device)
model_finetuned.load_state_dict(torch.load("best_model2.pth"))
model_finetuned.eval()

# pretrained model 갖고오기
model_pretrained6 = cheng2020_attn(pretrained=True, quality=6).to(device).eval()
model_pretrained5 = cheng2020_attn(pretrained=True, quality=5).to(device).eval()
model_pretrained4 = cheng2020_attn(pretrained=True, quality=4).to(device).eval()
model_pretrained3 = cheng2020_attn(pretrained=True, quality=3).to(device).eval()
model_pretrained2 = cheng2020_attn(pretrained=True, quality=2).to(device).eval()
model_pretrained1 = cheng2020_attn(pretrained=True, quality=1).to(device).eval()

# test data 갖고오기
transform = get_test_transform()
ori_transform = get_ori_transform()

_, _, test_set = get_datasets(data1024_filename, transform=transform)

_, _, _test_set = get_datasets(data1024_filename, transform=ori_transform)

test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
_test_loader = DataLoader(_test_set, batch_size=1, shuffle=False)


# data 모델에 인풋 & 아웃풋
# fits.CompImageHDU(data=data, compression_type='RICE_1', tile_shape=(512, 512), quantize_level=opt.quantization)

total_bits_pt6 = 0
total_bits_pt5 = 0
total_bits_pt4 = 0
total_bits_pt3 = 0
total_bits_pt2 = 0
total_bits_pt1 = 0
total_bits_ft = 0

total_psnr_pt6 = 0
total_psnr_pt5 = 0
total_psnr_pt4 = 0
total_psnr_pt3 = 0
total_psnr_pt2 = 0
total_psnr_pt1 = 0
total_psnr_ft = 0

total_mssim_pt6 = 0
total_mssim_pt5 = 0
total_mssim_pt4 = 0
total_mssim_pt3 = 0
total_mssim_pt2 = 0
total_mssim_pt1 = 0
total_mssim_ft = 0

i = 0 ###########

with torch.no_grad():
    for (input_pr, input_or) in tqdm(zip(test_loader, _test_loader)):
        i += 1
        input_or = input_or.to(device, non_blocking=True)
        input_pr = input_pr.to(device, non_blocking=True)

        # pretrained
        output_pt6 = model_pretrained6(input_pr)
        output_pt5 = model_pretrained5(input_pr)
        output_pt4 = model_pretrained4(input_pr)
        output_pt3 = model_pretrained3(input_pr)
        output_pt2 = model_pretrained2(input_pr)
        output_pt1 = model_pretrained1(input_pr)
        
        # finetuned
        output_ft = model_finetuned(input_pr) 

        pt6_xhat, pt6_likelihoods = output_pt6['x_hat'], output_pt6['likelihoods']
        pt5_xhat, pt5_likelihoods = output_pt5['x_hat'], output_pt5['likelihoods']
        pt4_xhat, pt4_likelihoods = output_pt4['x_hat'], output_pt4['likelihoods']
        pt3_xhat, pt3_likelihoods = output_pt3['x_hat'], output_pt3['likelihoods']
        pt2_xhat, pt2_likelihoods = output_pt2['x_hat'], output_pt2['likelihoods']
        pt1_xhat, pt1_likelihoods = output_pt1['x_hat'], output_pt1['likelihoods']
        ft_xhat, ft_likelihoods = output_ft['x_hat'], output_ft['likelihoods']
        # float32, [0,1]

        log_ori = torch.log1p(torch.clamp(input_or,min=0))
        maxv_log = log_ori.max()
        minv_log = log_ori.min()

        # unpreprocess
        pt6_unpr = unpreprocess(maxv_log, minv_log, pt6_xhat).cpu().numpy()
        pt5_unpr = unpreprocess(maxv_log, minv_log, pt5_xhat).cpu().numpy()
        pt4_unpr = unpreprocess(maxv_log, minv_log, pt4_xhat).cpu().numpy()
        pt3_unpr = unpreprocess(maxv_log, minv_log, pt3_xhat).cpu().numpy()
        pt2_unpr = unpreprocess(maxv_log, minv_log, pt2_xhat).cpu().numpy()
        pt1_unpr = unpreprocess(maxv_log, minv_log, pt1_xhat).cpu().numpy()
        ft_unpr = unpreprocess(maxv_log, minv_log, ft_xhat).cpu().numpy() # (1024, 1024) [0,32767]

        # input_or: (1,1,1024,1024) -> (1024,1024)
        input_or = input_or.squeeze().cpu().numpy()

        # cal bbp
        temp_pt6_bits_y = calculate_bits(pt6_likelihoods['y'])
        temp_pt6_bits_z = calculate_bits(pt6_likelihoods['z'])
        print(pt6_likelihoods['y'].shape, pt6_likelihoods['z'].shape)
        temp_pt6_bits = temp_pt6_bits_y + temp_pt6_bits_z
        temp_pt5_bits_y = calculate_bits(pt5_likelihoods['y'])
        temp_pt5_bits_z = calculate_bits(pt5_likelihoods['z'])
        temp_pt5_bits = temp_pt5_bits_y + temp_pt5_bits_z
        temp_pt4_bits_y = calculate_bits(pt4_likelihoods['y'])
        temp_pt4_bits_z = calculate_bits(pt4_likelihoods['z'])
        temp_pt4_bits = temp_pt4_bits_y + temp_pt4_bits_z
        temp_pt3_bits_y = calculate_bits(pt3_likelihoods['y'])
        temp_pt3_bits_z = calculate_bits(pt3_likelihoods['z'])
        temp_pt3_bits = temp_pt3_bits_y + temp_pt3_bits_z
        temp_pt2_bits_y = calculate_bits(pt2_likelihoods['y'])
        temp_pt2_bits_z = calculate_bits(pt2_likelihoods['z'])
        temp_pt2_bits = temp_pt2_bits_y + temp_pt2_bits_z
        temp_pt1_bits_y = calculate_bits(pt1_likelihoods['y'])
        temp_pt1_bits_z = calculate_bits(pt1_likelihoods['z'])
        temp_pt1_bits = temp_pt1_bits_y + temp_pt1_bits_z
        temp_ft_bits_y = calculate_bits(ft_likelihoods['y'])
        temp_ft_bits_z = calculate_bits(ft_likelihoods['z'])
        temp_ft_bits = temp_ft_bits_y + temp_ft_bits_z
       

        total_bits_pt6 += temp_pt6_bits
        total_bits_pt5 += temp_pt5_bits
        total_bits_pt4 += temp_pt4_bits
        total_bits_pt3 += temp_pt3_bits
        total_bits_pt2 += temp_pt2_bits
        total_bits_pt1 += temp_pt1_bits
        total_bits_ft += temp_ft_bits
        
        # cal psnr
        temp_pt6_psnr = calculate_psnr(input_or, pt6_unpr)
        temp_pt5_psnr = calculate_psnr(input_or, pt5_unpr)
        temp_pt4_psnr = calculate_psnr(input_or, pt4_unpr)
        temp_pt3_psnr = calculate_psnr(input_or, pt3_unpr)
        temp_pt2_psnr = calculate_psnr(input_or, pt2_unpr)
        temp_pt1_psnr = calculate_psnr(input_or, pt1_unpr)
        temp_ft_psnr = calculate_psnr(input_or, ft_unpr)

        total_psnr_pt6 += temp_pt6_psnr
        total_psnr_pt5 += temp_pt5_psnr
        total_psnr_pt4 += temp_pt4_psnr
        total_psnr_pt3 += temp_pt3_psnr
        total_psnr_pt2 += temp_pt2_psnr
        total_psnr_pt1 += temp_pt1_psnr
        total_psnr_ft += temp_ft_psnr

        # cal ms-ssim
        temp_pt6_mssim = calculate_ms_ssim(input_or, pt6_unpr)
        temp_pt5_mssim = calculate_ms_ssim(input_or, pt5_unpr)
        temp_pt4_mssim = calculate_ms_ssim(input_or, pt4_unpr)
        temp_pt3_mssim = calculate_ms_ssim(input_or, pt3_unpr)
        temp_pt2_mssim = calculate_ms_ssim(input_or, pt2_unpr)
        temp_pt1_mssim = calculate_ms_ssim(input_or, pt1_unpr)
        temp_ft_mssim = calculate_ms_ssim(input_or, ft_unpr)

        total_mssim_pt6 += temp_pt6_mssim
        total_mssim_pt5 += temp_pt5_mssim
        total_mssim_pt4 += temp_pt4_mssim
        total_mssim_pt3 += temp_pt3_mssim
        total_mssim_pt2 += temp_pt2_mssim
        total_mssim_pt1 += temp_pt1_mssim
        total_mssim_ft += temp_ft_mssim

        
        if i >= 2:
            break

        

# totla bits/ counts
img = next(iter(test_loader))
N,_,H,W = img.size()
print(N,"/",H,"/",W)
num_pixels = N*H*W

# count = len(test_loader)
count = i  ###################

pt6_bpp = total_bits_pt6/count/num_pixels
pt5_bpp = total_bits_pt5/count/num_pixels
pt4_bpp = total_bits_pt4/count/num_pixels
pt3_bpp = total_bits_pt3/count/num_pixels
pt2_bpp = total_bits_pt2/count/num_pixels
pt1_bpp = total_bits_pt1/count/num_pixels
ft_bpp = total_bits_ft/count/num_pixels

pt6_psnr = total_psnr_pt6/count
pt5_psnr = total_psnr_pt5/count
pt4_psnr = total_psnr_pt4/count
pt3_psnr = total_psnr_pt3/count
pt2_psnr = total_psnr_pt2/count
pt1_psnr = total_psnr_pt1/count
ft_psnr = total_psnr_ft/count

pt6_mssim = total_mssim_pt6/count
pt5_mssim = total_mssim_pt5/count
pt4_mssim = total_mssim_pt4/count
pt3_mssim = total_mssim_pt3/count
pt2_mssim = total_mssim_pt2/count
pt1_mssim = total_mssim_pt1/count
ft_mssim = total_mssim_ft/count

pt_bpp_lst = np.array([pt6_bpp, pt5_bpp, pt4_bpp, pt3_bpp, pt2_bpp, pt1_bpp])
pt_psnr_lst = np.array([pt6_psnr, pt5_psnr, pt4_psnr, pt3_psnr, pt2_psnr, pt1_psnr])
pt_mssim_lst = np.array([pt6_mssim, pt5_mssim, pt4_mssim, pt3_mssim, pt2_mssim, pt1_mssim])

print("pt_bpp: ",pt_bpp_lst)
print("pt_psnr: ",pt_psnr_lst)
print("pt_mssim: ",pt_mssim_lst)
print("ft_bpp: ",ft_bpp)
print("ft_psnr: ",ft_psnr)
print("ft_mssim: ",ft_mssim)

plt.figure(figsize=(8, 6))
plt.plot(pt_bpp_lst, pt_psnr_lst, 'red', linewidth=2, marker='o', markersize=8, label='Pretrained AI model')
plt.scatter(ft_bpp, ft_psnr, s=150, c='orange', marker='*', label='Fine-tuned AI model', zorder=2, edgecolor='black')
plt.xlabel('Bit-rate [bpp]', fontsize=20)
plt.ylabel('PSNR [dB]', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Rate-Distortion Curve', fontsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.savefig('RD_curve_psnr.png')

plt.figure(figsize=(8, 6))
plt.plot(pt_bpp_lst, pt_mssim_lst, 'red', linewidth=2, marker='o', markersize=8, label='Pretrained AI model')
plt.scatter(ft_bpp, ft_mssim, s=150, c='orange', marker='*', label='Fine-tuned AI model', zorder=2, edgecolor='black')
plt.xlabel('Bit-rate [bpp]', fontsize=20)
plt.ylabel('MS-SSIM', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Rate-Distortion Curve', fontsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.savefig('RD_curve_ms_ssim.png')

# watch -d -n 0.5 nvidia-smi

## 백그라운드에서 코드 실행, vscode 닫아도 ㄱㅊ
# CUDA_VISIBLE_DEVICES=2 nohup python -u test.py > log_test2.txt 2>&1 &

## 실시간 로그 확인
# tail -f log_test2.txt

# pkill -f test.py