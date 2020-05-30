from CreateDataset_02 import MNISTDataset, ToTensor
from CreatingModel_05 import Model
import torch
import torch.nn.functional as F


if __name__ == '__main__':
    train_dataset = MNISTDataset(
        'train.csv', 'Dataset/MNISTDataSet/train', transform=ToTensor())

    model = Model()
    print(model)
    torch.set_grad_enabled(False)
    sample = next(iter(train_dataset))
    image = sample['image']
    labels = sample['label']
    image_batch = image.unsqueeze(0)

    preds = model(image_batch)
    loss = F.cross_entropy(preds, labels)
    print(loss.item())
