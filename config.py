import torch

data1024_filename = 'data1024lst.txt'
PATH = "./data/CME_unpiped/"
PATH_171 = "./SDOML_171_zip/06/01/"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
batch_size = 16
learning_rate = 1e-8
patience = 100 # max patience
num_epochs = 500
num_workers = 8
train_loss_log_path = 'log/train_loss_list5.txt'
val_loss_log_path = 'log/val_loss_list5.txt'
train_loss_mse_log_path = 'log/train_loss_mse_list5.txt'
val_loss_mse_log_path = 'log/val_loss_mse_list5.txt'
train_loss_bpp_log_path = 'log/train_loss_bpp_list5.txt'
val_loss_bpp_log_path = 'log/val_loss_bpp_list5.txt'