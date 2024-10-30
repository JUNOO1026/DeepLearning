import torch
from torch import nn, optim
from torchvision import datasets, transforms

from model import *
from train import *
import yaml
import numpy as np
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


with open('hyperparameter.yaml') as f:
    params = yaml.safe_load(f)

BATCH_SIZE = params['BATCH_SIZE']
EPOCH = params['EPOCH']
TRAIN_RATIO = params['TRAIN_RATIO']
LR = float(params['LR'])
train_new_model = params['train_new_model']
LR_STEP = int(params['LR_STEP'])
LR_GAMMA = float(params['LR_GAMMA'])
SAVE_MODEL_PATH = params['SAVE_MODEL_PATH']
SAVE_HISTORY_PATH = params['SAVE_HISTORY_PATH']
criterion = nn.CrossEntropyLoss()

print(f"batch size : {BATCH_SIZE} \n"
      f"epoch : {EPOCH} \n"
      f"TRAIN RATION : {TRAIN_RATIO} \n"
      f"LR : {LR} \n"
      f"train_new Model : {train_new_model}")

train_transforms = transforms.ToTensor()
test_transforms = transforms.ToTensor()

train_DS = datasets.STL10(root='D:/DeepLearning_org/dataset/STL10', split='train', download=True, transform=train_transforms)
test_DS = datasets.STL10(root='D:\DeepLearning_org\dataset\STL10', split='test', download=True, transform=test_transforms)

NoT = int(len(train_DS) * TRAIN_RATIO)
NoV = len(train_DS) - NoT

train_DS, val_DS = torch.utils.data.random_split(train_DS, [NoT, NoV])


train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
val_DL = torch.utils.data.DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=True)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=False)


if train_new_model:
    model = CNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss, acc = Train(model, train_DL, val_DL, criterion, optimizer, EPOCH,
                      BATCH_SIZE, TRAIN_RATIO,
                      SAVE_MODEL_PATH, SAVE_HISTORY_PATH, LR_STEP=LR_STEP, LR_GAMMA=LR_GAMMA)

    print(len(loss['valid']))
    print(loss['valid'])
    plt.figure(figsize=(8, 12))
    plt.plot(range(1, EPOCH+1), loss['valid'])
    plt.xlabel('EPOCH')
    plt.ylabel('loss')
    plt.title('validation loss')
    plt.grid()
    plt.show()
