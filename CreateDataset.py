import pandas as pd
from pathlib import Path
import torch
from matplotlib import image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


class MNISTDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = Path(self.root_dir).joinpath(
            self.landmarks_frame['filename'].iloc[idx])
        my_image = image.imread(str(img_name))[:, :, 0]
        label = self.landmarks_frame['label'].iloc[idx]
        sample = {'image': my_image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


if __name__ == "__main__":
    dataset = MNISTDataset(
        'train.csv', 'Dataset/MNISTDataSet/train', transform=transforms.ToTensor())
    sample = dataset.__getitem__(0)
    print(sample['image'].shape, sample['label'])
