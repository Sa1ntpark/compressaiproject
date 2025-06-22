# main
# <전제>
# 논문에서의 전처리는 16bit정수형 -> 8bit정수형 byte scale
# 모델도 8bit정수형으로 양자화 되어야 하는가? 해석의 모호함 존재
# -> 우선 전처리에서 byte scale은 빼고 진행

# loss함수 정의가 모델의 입력 출력의 값 범위가 [0,255]인것이 전제이다.
# 그에 맞춰 모델의 입력 출력 데이터를 처리한다.
# -> 이렇게 하니 모델 출력이 발산하는 문제 발생
# 다시 입력 범위 [0,1]로 맞춤
# loss의 distortion에 곱해진 255^2을 없앰

# 학습속도 너무 느림, 1epoch가 2시간 걸림;
# 이미지 사이즈 256 by 256 으로 resize 함
# batch size = 1
from config import *
from compressai.losses import RateDistortionLoss
from dataset import get_datasets, read_loss_file, plot_dual_axis
from transforms import get_train_transform
from train import train
from compressai.zoo import cheng2020_attn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# quality=6 → 논문에서 사용한 lambda=0.0483에 해당
model = cheng2020_attn(pretrained=True, quality=6).to(device)
criterion = RateDistortionLoss(lmbda=0.0483)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
transform = get_train_transform()



# dataset
train_set, val_set, test_set = get_datasets(data1024_filename, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
# test_loader = DataLoader(test_set, shuffle=False, pin_memory=True)

train(model=model,train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, criterion=criterion)

# 저장된 loss list 가져오기
train_loss_lst = read_loss_file(train_loss_log_path)
val_loss_lst = read_loss_file(val_loss_log_path)
train_loss_mse_lst = read_loss_file(train_loss_mse_log_path)
val_loss_mse_lst = read_loss_file(val_loss_mse_log_path)
train_loss_bpp_lst = read_loss_file(train_loss_bpp_log_path)
val_loss_bpp_lst = read_loss_file(val_loss_bpp_log_path)

# loss list plot이미지 저장
plot_dual_axis(train_loss_lst, val_loss_lst, mode="all_loss")
plot_dual_axis(train_loss_mse_lst, val_loss_mse_lst, mode="mse_loss")
plot_dual_axis(train_loss_bpp_lst, val_loss_bpp_lst, mode="bpp_loss")

# watch -d -n 0.5 nvidia-smi
# CUDA_VISIBLE_DEVICES=2 python main.py

## 백그라운드에서 코드 실행, vscode 닫아도 ㄱㅊ
# CUDA_VISIBLE_DEVICES=2 nohup python -u main.py > log/log5.txt 2>&1 &

## 실시간 로그 확인
# tail -f log/log5.txt

## 실제로 프로그램 돌아가는 지 확인
# pgrep -u park_i python

## 실행중인 main.py 중지
# pkill -f main.py
