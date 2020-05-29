import pandas as pd
from pathlib import Path
import torch
from matplotlib import image
from torch.utils.data import Dataset
import numpy as np


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image = np.array(image)[:, :, 0]
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'image': torch.FloatTensor(image).unsqueeze(0),
                #         return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(np.array(label)).unsqueeze(0)}


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
        my_image = image.imread(str(img_name))
        label = self.landmarks_frame['label'].iloc[idx]
        sample = {'image': my_image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    dataset = MNISTDataset(
        'train.csv', 'Dataset/MNISTDataSet/train', transform=ToTensor())
    sample = dataset.__getitem__(0)
    print(sample['image'].shape, sample['label'])
