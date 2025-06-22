from config import *
import time
from dataset import save_loss_to_txt

def train(model, train_loader, val_loader, optimizer, criterion):
      
    print("train start")
    j = 0
    train_data_num =len(train_loader.dataset)
    val_data_num =len(val_loader.dataset)
    best_val_loss = 100
    t0 = time.time()
    model.train()  # 학습 모드로 설정
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_loss_bpp = 0.0
        train_loss_mse = 0.0
        # t1 = time.time()
        for i, inputs in enumerate(train_loader):
            # 순전파 (Forward pass)
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            # 역전파 및 파라미터 업데이트
            optimizer.zero_grad()
            loss['loss'].backward() # gradient 계산
            optimizer.step() # 모델 업데이트
            train_loss += loss['loss'] * inputs.size(0)
            train_loss_bpp += loss['bpp_loss'] * inputs.size(0)
            train_loss_mse += loss['mse_loss'] * inputs.size(0)
            if i%100 == 0:
                    t2 = time.time()
                    elapsed_seconds = int(t2 - t0)
                    hours, remainder = divmod(elapsed_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)    
                    print(f"epoch[{epoch + 1}] | iteration[{i}] / ratio:{(i+1)/len(train_loader)*100:.3f}% / train_loss:{train_loss/(batch_size*(i+1)):.4e} / loss_mse:{train_loss_mse/(batch_size*(i+1)):.4e} / loss_bpp:{train_loss_bpp/(batch_size*(i+1)):.4e} / cumulative time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        train_loss /= train_data_num
        train_loss_bpp /= train_data_num
        train_loss_mse /= train_data_num  
        if train_loss == torch.nan:
            raise ValueError("NaN detected in train_loss, check your model or data preprocessing.")   
        # train_loss_lst.append(train_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}] / train_loss: {train_loss:.4e}')

        # 검증 루프
        model.eval()  # 평가 모드로 설정
        val_loss = 0.0
        val_loss_bpp = 0.0
        val_loss_mse = 0.0
        with torch.no_grad():
            for inputs_val in val_loader:
                inputs_val = inputs_val.to(device, non_blocking=True)
                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, inputs_val)
                val_loss += loss_val['loss'] * inputs_val.size(0)
                val_loss_bpp += loss_val['bpp_loss'] * inputs_val.size(0)
                val_loss_mse += loss_val['mse_loss'] * inputs_val.size(0)
            val_loss /= val_data_num
            val_loss_bpp /= val_data_num
            val_loss_mse /= val_data_num
            # val_loss_lst.append(val_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4e}, Validation BPP: {val_loss_bpp:.4e}, Validation MSE: {val_loss_mse:.4e}')

            # Early Stopping 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                j = 0
                print(f"Epoch[{epoch+1}] best model updated")
                torch.save(model.state_dict(), f'best_model5.pth') # Best 모델 저장
            else:
                j += 1
                if j >= patience:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                    break
        # train_loss_lst, val_loss_lst 파일로 저장
        save_loss_to_txt(train_loss, val_loss, 
                         train_loss_mse, val_loss_mse,
                         train_loss_bpp, val_loss_bpp)

        t4 = time.time()
        elapsed_seconds = int(t4 - t0)
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"one epoch done, cummulative time: {hours:02d}:{minutes:02d}:{seconds:02d} ")

        