import os
import yaml

import pandas as pd

import torch
import xml.etree.ElementTree as ET

from PIL import Image

yaml_path = '../../model_yaml/yolov1/yolov1.yaml'


def load_yaml_dataset(path):
    with open(yaml_path, 'r') as f:
        yolov1_info = yaml.safe_load(f)

    train_path = yolov1_info['train_dir']
    test_path = yolov1_info['test_dir']

    return train_path, test_path


train_path, test_path = load_yaml_dataset(yaml_path)


def load_class(path):
    with open(yaml_path, 'r') as f:
        yolov1_info = yaml.safe_load(f)

    label_class = yolov1_info['class']

    return label_class


print(load_class(yaml_path))


def image_annot_df(path, image_name, annot_name):
    images = [images for images in sorted(os.listdir(path)) if images.endswith('.jpg')]
    annots = [annots for annots in sorted(os.listdir(path)) if annots.endswith('.xml')]

    images_series = pd.Series(images, name=image_name)
    annots_series = pd.Series(annots, name=annot_name)
    df = pd.concat([images_series, annots_series], axis=1)

    return pd.DataFrame(df)


train_df = image_annot_df(train_path, 'train_images', 'train_annots')
test_df = image_annot_df(test_path, 'test_images', 'test_annots')

# print(train_df)
print(train_df.iloc[0, 0])


class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, df, files_dir, S=7, B=2, C=3, transform=None):
        self.annotation = df
        self.files_dir = files_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        label_path = os.path.join(self.files_dir, self.annotation.iloc[idx, 1])
        boxes = []
        tree = ET.parse(label_path)
        root = tree.getroot()

        class_dict = {cls: idx for idx, cls in enumerate(load_class(yaml_path))}

        if int(root.find('size').find('height').text) == 0:
            filename = root.find('filename').text
            image = Image.open(self.files_dir + '/' + filename)
            img_width, img_height = image.size

            for member in root.findall('object'):
                obj_name = member.find('name').text
                obj_class_map = class_dict[obj_name]

                # bounding box
                xmin = int(member.find('bndbox').find('xmin').text)
                ymin = int(member.find('bndbox').find('ymin').text)
                xmax = int(member.find('bndbox').find('xmax').text)
                ymax = int(member.find('bndbox').find('ymax').text)

                center_x = ((xmax + xmin) / 2) / img_width
                center_y = ((ymax + ymin) / 2) / img_height
                box_width = (xmax - xmin) / img_width  # 0과 1사이로 됨
                box_height = (ymax - ymin) / img_height

                boxes.append([obj_class_map, center_x, center_y, box_width, box_height])

        elif int(root.find('size').find('height').text) != 0:
            for member in root.findall('object'):
                obj_name = member.find('na  me').text
                obj_class_map = class_dict[obj_name]

                # bounding box
                xmin = int(member.find('bndbox').find('xmin').text)
                ymin = int(member.find('bndbox').find('ymin').text)
                xmax = int(member.find('bndbox').find('xmax').text)
                ymax = int(member.find('bndbox').find('ymax').text)

                img_width = int(root.find('size').find('width').text)
                img_height = int(root.find('size').find('height').text)

                center_x = ((xmax + xmin) / 2) / img_width
                center_y = ((ymax + ymin) / 2) / img_height
                box_width = (xmax - xmin) / img_width
                box_height = (ymax - ymin) / img_height

                boxes.append([obj_class_map, center_x, center_y, box_width, box_height])

        boxes = torch.tensor(boxes)
        img_path = os.path.join(self.files_dir, self.annotation.iloc[idx, 0])
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)  # 7*7 그리드에서 어느 셀에 있는지
            x_cell, y_cell = self.S * x - j, self.S * y - i  # 셀 내부에서 바운딩 박스 중심의 상대 좌표가 어딘지

            width_cell, height_cell = width * self.S, height * self.S

            for b in range(self.B):
                if label_matrix[i, j, self.C + b * 5] == 0:
                    label_matrix[i, j, self.C + b * 5] = 1

                    start_idx = self.C + b * 5 + 1  # 4, 9
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    label_matrix[i, j, start_idx:start_idx + 4] = box_coordinates

                    label_matrix[i, j, class_label] = 1

        return image, label_matrix
