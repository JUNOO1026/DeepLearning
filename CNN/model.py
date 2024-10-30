import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU()) # 이미지 사이즈가 줄지 않음.
        self.maxpool1 = nn.MaxPool2d(2)  # (32, 16, 48, 48)
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
        self.maxpool2 = nn.MaxPool2d(2)  # (32, 32, 24, 24)
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.averagepool1 = nn.AvgPool2d(2) # (32, 64, 12, 12)
        self.layer4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.averagepool2 = nn.AvgPool2d(2) # (32, 128, 6, 6)
        self.fc = nn.Linear(128*6*6, 10)


    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool1(x)
        x = self.layer2(x)
        x = self.maxpool2(x)
        x = self.layer3(x)
        x = self.averagepool1(x)
        x = self.layer4(x)
        x = self.averagepool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

# model = CNN()
# a = torch.randn(32, 3, 96, 96)
# print(model(a).shape)