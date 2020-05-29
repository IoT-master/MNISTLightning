import torch.nn as nn
import torch.nn.functional as F
import torch

import platform
from torch.utils.data import DataLoader
from CreateDataset_02 import MNISTDataset
from torchvision import transforms


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        # [b, 1, 28, 28] ==> [b, 20, 24, 24]
        x = F.relu(self.conv1(x))
        # [b, 20, 24, 24] ==> [b, 20, 12, 12]
        x = F.max_pool2d(x, 2, 2)
        # [b, 20, 12, 12] ==> [b, 50, 8, 8]
        x = F.relu(self.conv2(x))
        # [b, 50, 8, 8] ==> [b, 50, 4, 4]
        x = F.max_pool2d(x, 2, 2)
        # [b, 50, 4, 4] ==> [b, 50*4*4]
        x = x.view(-1, 4*4*50)
        # [b, 50*4*4] ==> [b, 60]
        x = F.relu(self.fc1(x))
        # [b, 60] ==> [b, 10]
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    train_dataset = MNISTDataset(
        'train.csv', 'Dataset/MNISTDataSet/train', transform=transforms.ToTensor())

    model = Model()
    torch.set_grad_enabled(False)
    sample = next(iter(train_dataset))
    image = sample['image']
    label = sample['label']
    image_batch = image.unsqueeze(0)
    print(model(image_batch).shape)
