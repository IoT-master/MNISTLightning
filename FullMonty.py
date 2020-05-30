import torch.optim as optim
import torch.nn as nn
import torch
from pathlib import Path
from matplotlib import image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


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


train_dataset = MNISTDataset(
    'train.csv', 'Dataset/MNISTDataSet/train', transform=ToTensor())


batch_loader_params = {
    "batch_size": 25,
    "shuffle": True,
    "num_workers": 2
}
train_batches = DataLoader(train_dataset, **batch_loader_params)


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


model = Model()
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.01)

total_loss = 0
total_correct = 0

for epoch in range(5):
    for batch in train_batches:
        images = batch['image']
        labels = batch['label']

        preds = model(images)  # Pass Batch
        loss = F.cross_entropy(preds, labels)  # Calculate Loss

        optimizer.zero_grad()  # PyTorch Adds to this, so you have to zero it out after one batch
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print("epoch:", epoch, "total_correct:",
          total_correct, "loss:", total_loss)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

torch.save(model.state_dict(), 'MNISTModel.pt')

# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
