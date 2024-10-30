import torch
from torch import optim

from tqdm import tqdm

from model_code.yolov1.yolov1 import YoloV1
from dataset_code.yolov1.yolov1_dataset import FruitDataset


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        loss = loss_fn(pred, y)
        mean_loss.append(loss.ltem())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")


def main():
    model = YoloV1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=)



