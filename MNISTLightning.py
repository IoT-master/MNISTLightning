from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import functional as F
from torch import nn
from torch import optim
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
from matplotlib import image


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
                'label': torch.from_numpy(np.array(label))}


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


class LitModel(LightningModule):
    def __init__(self):
        super(LitModel, self).__init__()
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

    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']

        preds = self(images)
        loss = F.cross_entropy(preds, labels)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)

    def train_dataloader(self):
        train_dataset = MNISTDataset(
            'train.csv', 'Dataset/MNISTDataSet/train', transform=ToTensor())

        batch_loader_params = {
            "batch_size": 25,
            "shuffle": True,
            "num_workers": 4
        }
        train_batches = DataLoader(train_dataset, **batch_loader_params)
        return train_batches


model = LitModel()

# most basic trainer, uses good defaults
# trainer = Trainer(gpus=8, num_nodes=1)
trainer = Trainer()
trainer.fit(model)
torch.save(model.state_dict(), 'MNISTModelLIB.pt')
