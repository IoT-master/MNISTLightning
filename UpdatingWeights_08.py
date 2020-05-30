from CreateDataset_02 import MNISTDataset, ToTensor
from CreatingModel_05 import Model
import torch
import torch.nn.functional as F
from torch import optim


if __name__ == '__main__':
    train_dataset = MNISTDataset(
        'train.csv', 'Dataset/MNISTDataSet/train', transform=ToTensor())

    def get_num_correct(preds, labels):
        print(preds, labels)
        return preds.argmax(dim=1).eq(labels).sum().item()

    model = Model()
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    sample = next(iter(train_dataset))
    image = sample['image']
    labels = sample['label']
    print(labels)
    image_batch = image.unsqueeze(0)

    preds = model(image_batch)
    loss = F.cross_entropy(preds, labels.unsqueeze(0))
    # print(model.conv1.weight.grad)
    loss.backward()
    print(model.conv1.weight.grad.shape)
    # conv1 has a shape of [20, 1, 5, 5]
    # takes 20 inputs, one kernel (one channel) output that's 5x5 in size
    print(loss.item())

    print(get_num_correct(preds, labels))

    optimizer.step()
    optimizer.zero_grad()

    preds = model(image_batch)
    loss = F.cross_entropy(preds, labels)
    loss.backward()
    print(model.conv1.weight.grad.shape)
    print(loss.item())

    print(get_num_correct(preds, labels))

    optimizer.step()
    optimizer.zero_grad()

    preds = model(image_batch)
    loss = F.cross_entropy(preds, labels)
    loss.backward()
    print(model.conv1.weight.grad.shape)
    print(loss.item())
    optimizer.zero_grad()
    print(get_num_correct(preds, labels))

    optimizer.step()
    optimizer.zero_grad()

    preds = model(image_batch)
    loss = F.cross_entropy(preds, labels)
    loss.backward()
    print(model.conv1.weight.grad.shape)
    print(loss.item())
    optimizer.zero_grad()
    print(get_num_correct(preds, labels))

    optimizer.step()
    optimizer.zero_grad()

    preds = model(image_batch)
    loss = F.cross_entropy(preds, labels)
    loss.backward()
    print(model.conv1.weight.grad.shape)
    print(loss.item())
    optimizer.zero_grad()
    print(get_num_correct(preds, labels))

    optimizer.step()
    optimizer.zero_grad()
