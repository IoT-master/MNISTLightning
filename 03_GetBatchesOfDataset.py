import platform
from torch.utils.data import DataLoader
from CreateDataset import MNISTDataset
from torchvision import transforms

train_dataset = MNISTDataset(
    'train.csv', 'Dataset/MNISTDataSet/train', transform=transforms.ToTensor())

batch_loader_params = {
    "batch_size": 50,
    "shuffle": True,
    "num_workers": 0 if platform.system() == 'Windows' else 2
}
dataloader = DataLoader(train_dataset, **batch_loader_params)

if __name__ == '__main__':
    sample = iter(dataloader)
    print(sample.next()['label'], sample.next()['image'].shape)
