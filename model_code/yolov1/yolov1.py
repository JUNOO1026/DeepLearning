import os
import yaml

import torch
from torch import nn

current_path = os.path.dirname(__file__)
print("current_path : ", current_path)

yaml_path = '../../model_yaml/yolov1/yolov1.yaml'

def load_yaml_model(path):
    with open(path, 'r') as f:
        yolov1_info = yaml.safe_load(f)

    architecture_config = yolov1_info['architecture_config']

    return architecture_config

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.batchnorm(self.conv1(x)))

class YoloV1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()

        self.architecture = load_yaml_model(yaml_path)
        self.in_channels = in_channels
        self.darknet = self._create_block(self.architecture)
        self.fcs = self._create_fcs(**kwargs)


    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))


    def _create_block(self, architecture):
        layer = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, list) and len(x) == 4:
                layer += [CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]
            elif isinstance(x, str):
                layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif isinstance(x, list) and len(x) == 3:
                conv1 = x[0]
                conv2 = x[1]
                repeat = x[2]
                for i in range(repeat):
                    layer += [CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])]
                    layer += [CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])]
                    in_channels = conv2[1]

        return nn.Sequential(*layer)


    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(nn.Flatten(),
                             nn.Linear(S * S * 1024, 4096),
                             nn.Dropout(0.2),
                             nn.LeakyReLU(0.1),
                             nn.Linear(4096, S * S * (B * 5 + C)))



x = torch.randn(4, 3, 448, 448)

model = YoloV1(split_size=7, num_boxes=2, num_classes=3)

print(model(x).shape)


