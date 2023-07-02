import torch
import torch.nn as nn

# https://stackoverflow.com/questions/73147887/problem-implementing-a-vgg11-from-scratch-in-pytorch

class VGG(nn.Module):
    def __init__(self,num_classes):
        super(VGG, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(100, 10)
        )

    # the forward pass
    def forward(self, x):
        # print(x.shape)
        out = self.layer(x)
        # print(out.shape)
        _out = out.view(out.size(0), -1)
        # print(_out.shape)
        _out = self.fc(_out)
        return _out
