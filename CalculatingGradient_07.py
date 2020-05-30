from CreateDataset_02 import MNISTDataset, ToTensor
from CreatingModel_05 import Model
import torch
import torch.nn.functional as F


if __name__ == '__main__':
    train_dataset = MNISTDataset(
        'train.csv', 'Dataset/MNISTDataSet/train', transform=ToTensor())

    model = Model()
    print(model)

    sample = next(iter(train_dataset))
    image = sample['image']
    labels = sample['label']
    image_batch = image.unsqueeze(0)

    preds = model(image_batch)
    loss = F.cross_entropy(preds, labels)
    print(loss.item())

    print(model.conv1.weight.grad)
    loss.backward()
    print(model.conv1.weight.grad.shape)
    # conv1 has a shape of [20, 1, 5, 5]
    # takes 20 inputs, one kernel (one channel) output that's 5x5 in size
