from torch.utils.data import DataLoader
import platform
import torch
from CreateDataset_02 import MNISTDataset, ToTensor
from CreatingModel_05 import Model
import torch.nn.functional as F
from torch import optim


torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


train_dataset = MNISTDataset(
    'train.csv', 'Dataset/MNISTDataSet/train', transform=ToTensor())

batch_loader_params = {
    "batch_size": 25,
    "shuffle": True,
    "num_workers": 0
}
train_batches = DataLoader(train_dataset, **batch_loader_params)

model = Model()
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.01)

total_loss = 0
total_correct = 0

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

print("epoch:", 0, "total_correct:", total_correct, "loss:", total_loss)
