import os

import pandas as pd
import torch
import torchvision.transforms as aug
from PIL import Image
from torch.utils.data import Dataset, DataLoader

train_img_foler = '../datasets/train/images'
test_img_folder = '../datasets/test/images'
data_csv_path = '../info/data.csv'


def read_train_data_xray():
    data = pd.read_csv('../info/train.csv')
    imgs = data['file_name']
    labels = data.drop('file_name', axis=1)
    return imgs, labels


class MyData(Dataset):
    def __init__(self):
        self.features, self.labels = read_train_data_xray()
        self.transforms = aug.Compose([aug.Resize((256, 256)), aug.ToTensor()])

    def __getitem__(self, index):
        path = os.path.join(train_img_foler, self.features[index])
        feature = Image.open(path)
        feature = self.transforms(feature)
        label = self.labels.iloc[index]
        labels = torch.tensor(label)
        return feature, labels

    def __len__(self):
        return len(self.labels)


def load_data_xray(batch_size):
    train_iter = DataLoader(MyData(), batch_size, shuffle=True)
    return train_iter


if __name__ == "__main__":
    read_train_data_xray()
