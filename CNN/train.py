import time
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def Train(model, train_DL, val_DL, criterion, optimizer, EPOCH,
          BATCH_SIZE, TRAIN_RATIO,
          save_model_path, save_history_path, **kwargs):

    if "LR_STEP" in kwargs:
        scheduler = StepLR(optimizer, step_size=kwargs['LR_STEP'], gamma=kwargs['LR_GAMMA'])
    else:
        scheduler = None

    loss_history = {'train': [], 'valid': []}
    acc_history = {'train': [], 'valid': []}
    best_loss = torch.inf

    for ep in range(EPOCH):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"epoch : {ep+1}, current_LR : {current_lr}")
        model.train() # 훈련 모드
        train_loss, train_acc, _ = loss_epoch(model, train_DL, criterion, optimizer)
        loss_history['train'].append(train_loss)
        acc_history['train'].append(train_acc)

        model.eval()
        with torch.no_grad(): # 기울기 계산이 필요없음. 성능 효율을 높이고 메모리를 아낄 수 있음
            val_loss, val_acc, _ = loss_epoch(model, val_DL, criterion)
            loss_history['valid'].append(val_loss)
            acc_history['valid'].append(val_acc)

            if val_loss < best_loss: # validation을 위주로 모델을 평가하니 best_loss보다 낮으면 그 모델을 저장해야한다.
                best_loss = val_loss
                torch.save({'model': model,
                            "ep": ep,
                            "optimizer": optimizer,
                            "scheduler": scheduler}, save_model_path + 'best.pt')

        if "LR_STEP" in kwargs:
            scheduler.step() # 지정한 epoch이 도달하면 scheduler 업데이트로 learning rate가 조절됨.

        print(f"train_loss: {round(train_loss, 5)},"
              f"valid_loss : {round(val_loss, 5)},"
              f"train_acc : {round(train_acc, 1)} %,"
              f"valid_acc : {round(val_acc, 1)}%",
              f"time : {round(time.time() - start_time)}s")
        print('-' * 30)

        torch.save({"loss_history": loss_history,
                    "acc_history": acc_history,
                    "EPOCH": EPOCH,
                    "BATCH_SIZE": BATCH_SIZE,
                    "TRAIN_RATIO":TRAIN_RATIO}, save_history_path + 'weight.pt')

    return loss_history, acc_history

def loss_epoch(model, DL, criterion, optimizer=None):
    r_loss = 0
    r_corrects = 0
    NoV = len(DL.dataset)  # Number of Validation data
    # model.eval()
    for x_batch, y_batch in tqdm(DL):
        x_batch = x_batch.to(DEVICE)  # to(DEVICE)를 하지 않으면 오류가 나긴하는데, GPU에서 학습을 하기 위한 단계라고 봐야함.
        y_batch = y_batch.to(DEVICE)  # x_batch, y_batch 모두 GPU단으로 올라가야 정상적으로 돌아갈 수 있기 때문임.
        y_hat = model(x_batch)

        loss = criterion(y_hat, y_batch)  # nn.CrossEntropyLoss() -> 배치 크기마다 나눠버리니
        if optimizer is not None:
            optimizer.zero_grad()  # 기울기 값을 초기화 함.
            loss.backward()  # BackPropagation 적용
            optimizer.step()  # weight update

        loss_b = loss.item() * x_batch.shape[0]  # nn.CrossEntropyLoss를 통해서 Update되는데, 이때 Softmax가 들어있기 때문에,
        r_loss += loss_b  # batch 단위로 나눠지게 되어 있음. 계산에 용이하기 위해 x_batch.shape[0]을 곱해줌.

        pred = y_hat.argmax(dim=1)
        corrects_b = torch.sum(pred == y_batch).item()
        r_corrects += corrects_b

    loss_e = r_loss / NoV
    accuracy_e = (r_corrects / NoV) * 100

    return loss_e, accuracy_e, r_corrects


# def loss_plot()