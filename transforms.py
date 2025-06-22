from torchvision import transforms
import torch
import numpy as np

def byte_scale(img): # torch.tensor
    img_min = img.min()
    img_max = img.max()
    scaled_img = (img - img_min) / (img_max - img_min) * 255.0
    return torch.clamp(scaled_img, 0, 255).to(torch.uint8)

def change_byte_order(img): # numpy array
    return img.astype(img.dtype.newbyteorder('='))

def clamp_0(img): # numpy array
    return np.clip(img,min= 0)  # 0 이하 값 처리

def divide_by_255(img): # numpy array
    return img / 255.0  # [0,255] -> [0,1] float32

def make_3_channel(img): # numpy array
    return torch.Tensor.repeat(img,3,1,1)

def to_numpy(img):
    return img.cpu().numpy()

def get_train_transform(): # resize to 256 by 256
    return transforms.Compose([
        # array to tensor
        # transform.resize 256 by 256
        # 0 이하 값 처리
        # log1p
        # bytescale [0,1] -> [0,255]
        # minmaxscaler [0,255] -> [0,1]
        # copy 3 channel

        change_byte_order, # numpy array byte order change
        transforms.ToTensor(), # (1024, 1024) -> (1, 1024, 1024) [minv,maxv]
        transforms.Resize((512,512)), # (1, 512, 512)
        clamp_0, # [0,maxv]
        torch.log1p, # log conversion [log_min, log_max]
        byte_scale, # [0,255] byte scale
        divide_by_255, # [0,255] -> [0,1] float32
        make_3_channel
    ])  # (3, 512, 512) 

def get_test_transform(): 
    return transforms.Compose([
        change_byte_order, # numpy array byte order change
        transforms.ToTensor(), # (1024, 1024) -> (1, 1024, 1024) [minv,maxv]
        clamp_0, # [0,maxv]
        torch.log1p, # log conversion [log_min, log_max]
        byte_scale, # [0,255] byte scale
        divide_by_255, # [0,255] -> [0,1] float32
        make_3_channel
        # (3, 1024, 1024) [0,1] float32
    ])  


# inverse transform
def get_ori_transform(): # resize to 512 by 512
    return transforms.Compose([
        change_byte_order,
        transforms.ToTensor(), 
        transforms.Resize((512,512)),
        clamp_0
    ])

def get_con_transform(): # resize to 512 by 512
    return transforms.Compose([
        change_byte_order,
        transforms.ToTensor(), 
        transforms.Resize((512,512)),
    ])
