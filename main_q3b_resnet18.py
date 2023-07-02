import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime
from torchvision import models as _models

from mac_config import device
from models.alexnet_q3 import AlexNetQ3


def get_train_valid_loader(
        data_dir, batch_size, augment, random_seed, valid_size=0.1, shuffle=True
):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    num_train = len(train_dataset)

    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler
    )

    return (train_loader, valid_loader)


def get_test_loader(data_dir, batch_size, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose(
        [
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader


# CIFAR10 dataset
train_loader, valid_loader = get_train_valid_loader(
    data_dir="./data", batch_size=64, augment=False, random_seed=1
)

test_loader = get_test_loader(data_dir="./data", batch_size=64)


if __name__ == '__main__':
    # +++++++++++++++++++++++++++++++++++++
    imagenet_num_classes = 1000
    cifar_num_classes = 10
    # +++++++++++++++++++++++++++++++++++++

    num_epochs = 20
    batch_size = 64
    learning_rate = 0.005

    model = _models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 10)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('{} - Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(str(datetime.now()), epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('[Base] Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('[Base] Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))
