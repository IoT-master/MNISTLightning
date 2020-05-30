from torch.utils.data import DataLoader
import platform
import torch
from CreateDataset_02 import MNISTDataset, ToTensor
from CreatingModel_05 import Model


# torch.set_printoptions(linewidth=120)
# torch.set_grad_enabled(True)

# train_dataset = MNISTDataset(
#     'train.csv', 'Dataset/MNISTDataSet/train', transform=ToTensor())

# batch_loader_params = {
#     "batch_size": 50,
#     "shuffle": True,
#     "num_workers": 0 if platform.system() == 'Windows' else 2
# }
# train_batches = DataLoader(train_dataset, **batch_loader_params)

model = Model()
print(model)
