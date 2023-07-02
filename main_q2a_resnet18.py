import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn

from mac_config import device
from models.alexnet import AlexNet
from cutout import Cutout
from models.googlenet import GoogLeNet
from models.resnet18 import ResNet, BasicBlock
from models.vgg11 import VGG
from train_and_test import train, test


MEAN = [0.4913, 0.4821, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def CIFAR10_data_transforms():
    MEAN = [0.4913, 0.4821, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(MEAN, STD)
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    return train_transform, valid_transform


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Settings of Hyperparameters
    batch_size = 32
    n_class = 10
    learning_rate = 1e-3

    momentum = 0.9
    weight_decay = 5e-4

    # Data Preparation
    data_dir = './'
    train_transform, valid_transform = CIFAR10_data_transforms()
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(data_dir, 'cifar10'), train=True, download=True,
                                            transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                               num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=os.path.join(data_dir, 'cifar10'), train=False, download=True,
                                           transform=valid_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                              num_workers=2)
    # Model Selection

    model_name = 'resnet18'
    model = ResNet(n_class, BasicBlock).to(device)

    # Criterion & Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    # Running Epoch
    num_epoch = 50
    train_accuracy_list = []
    test_accuracy_list = []
    train_loss_list = []
    test_loss_list = []

    for epoch in range(num_epoch):
        train_loss, train_accuracy = train(epoch, model, train_loader, criterion, optimizer, scheduler,device)

        test_loss, test_accuracy = test(model, test_loader, criterion, epoch,device)

        train_accuracy_list.append(train_accuracy / 100)
        test_accuracy_list.append(test_accuracy / 100)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        print(
            f'[{model_name}] finish epoch {epoch + 1}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, '
            f'train acc: {train_accuracy:.4f}% , test acc: {test_accuracy:.4f}% ')

    # summarize history for accuracy

    # save the trained model to disk
    torch.save({
        'epoch': num_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, f'./outputs/{model_name}_model.pth')

    plt.plot(range(num_epoch), train_accuracy_list, '-r')
    plt.plot(range(num_epoch), test_accuracy_list, '-g')

    plt.title(f'{model_name} Training statistics')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper left')
    plt.savefig(f'./outputs/{model_name}_accuracy.jpg')
    plt.show()

    plt.plot(range(num_epoch), train_loss_list, '-r')
    plt.plot(range(num_epoch), test_loss_list, '-g')
    plt.title(f'{model_name} Training Loss statistics')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['Train Loss', 'Test Loss'], loc='upper left')
    plt.savefig(f'./outputs/{model_name}_loss.jpg')
    plt.show()
