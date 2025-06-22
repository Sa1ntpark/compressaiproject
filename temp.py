# pretrained model과 finetuned model의 결과를 비교하는 코드입니다.
from dataset import get_datasets
from transforms import get_test_transform, get_ori_transform, get_train_transform, get_temp_transform
from compressai.zoo import cheng2020_attn
from torch.utils.data import DataLoader
# from config import *
import torch
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from custom_losses import CustomRateDistortionLoss

def unpreprocess(maxv_log, minv_log, res_img):
    res_img = res_img[0][0] # (B,3,H,W) -> (H,W)
    res_img = res_img * (maxv_log - minv_log) + minv_log
    res_img = torch.expm1(res_img)
    return res_img


test_transform = get_temp_transform()
ori_transform = get_ori_transform()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = CustomRateDistortionLoss(lmbda=0.0483)


model_pt = cheng2020_attn(pretrained=True, quality=6).to(device).eval()

model_ft = cheng2020_attn(pretrained=False, quality=6).to(device)
model_ft.load_state_dict(torch.load("best_model.pth"))
model_ft.eval()

_, _, test_set = get_datasets(transform=test_transform)
_, _, _test_set = get_datasets(transform=ori_transform)
_test_loader = DataLoader(_test_set, batch_size=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

data = next(iter(test_loader))
_data = next(iter(_test_loader))

data = data.to(device, non_blocking=True)

# model에 인풋 & 아웃풋
print("max data: %.3e"%data.max())
output_pt = model_pt(data)
output_ft = model_ft(data)
xhat_pt = output_pt['x_hat']
xhat_ft = output_ft['x_hat']
print("max xhat_pt: %.3e"%xhat_pt.max())
print("max xhat_ft: %.3e"%xhat_ft.max())
print("min xhat_pt: %.3e"%xhat_pt.min())
print("min xhat_ft: %.3e"%xhat_ft.min())
# loss 계산
# loss_pt = criterion(output_pt, data)
# loss_ft = criterion(output_ft, data)
# print("loss_pt: %.3e"%loss_pt['loss'].item())
# print("loss_ft: %.3e"%loss_ft['loss'].item())
# print("loss_pt bpp: %.3e"%loss_pt['bpp_loss'].item())
# print("loss_ft bpp: %.3e"%loss_ft['bpp_loss'].item())
# print("loss_pt mse: %.3e"%loss_pt['mse_loss'].item())
# print("loss_ft mse: %.3e"%loss_ft['mse_loss'].item())

output_pt = torch.clamp(xhat_pt, 0, 1)
output_ft = torch.clamp(xhat_ft, 0, 1)


# 원본 데이터 log max, min 구하기
_data_ori = torch.log1p(torch.clamp(_data,min=0))
log_maxv = _data_ori.max()
log_minv = _data_ori.min()

# 데이터 unpreprocess
data_pt = unpreprocess(log_maxv, log_minv, output_pt)
data_ft = unpreprocess(log_maxv, log_minv, output_ft)
data_pt = data_pt.detach().cpu().numpy()
data_ft = data_ft.detach().cpu().numpy()

plt.figure()
plt.imshow(data_pt, origin='lower', cmap='gray')  # 'gray'는 흑백 컬러맵, origin='lower'는 이미지 축 방향 설정
plt.colorbar(label='Intensity')  # 컬러바 추가 (데이터 값의 강도 표시)
plt.title('Pretrained model Image')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
plt.savefig('output_pt.png')


plt.figure()
plt.imshow(data_ft, origin='lower', cmap='gray')  # 'gray'는 흑백 컬러맵, origin='lower'는 이미지 축 방향 설정
plt.colorbar(label='Intensity')  # 컬러바 추가 (데이터 값의 강도 표시)
plt.title('Finetuned model Image')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
plt.savefig('output_ft.png')


plt.figure()
plt.imshow(_data[0][0], origin='lower', cmap='gray')  # 'gray'는 흑백 컬러맵, origin='lower'는 이미지 축 방향 설정
plt.colorbar(label='Intensity')  # 컬러바 추가 (데이터 값의 강도 표시)
plt.title('original Image')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
plt.savefig('original_image.png')

# watch -d -n 0.5 nvidia-smi

# CUDA_VISIBLE_DEVICES=1 python temp.py 