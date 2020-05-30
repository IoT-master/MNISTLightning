from torch.utils.data import DataLoader
import platform
import torch
from CreateDataset_02 import MNISTDataset, ToTensor


torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

train_dataset = MNISTDataset(
    'train.csv', 'Dataset/MNISTDataSet/train', transform=ToTensor())

batch_loader_params = {
    "batch_size": 50,
    "shuffle": True,
    "num_workers": 0 if platform.system() == 'Windows' else 2
}
train_batches = DataLoader(train_dataset, **batch_loader_params)

if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt
    import numpy as np

    batch_samples = iter(train_batches)
    samples = batch_samples.next()
    print(samples['image'].shape)
    datset_batch = torchvision.utils.make_grid(samples['image'])

    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    print(samples['label'])
    imshow(torchvision.utils.make_grid(datset_batch))
