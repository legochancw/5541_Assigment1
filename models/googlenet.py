import torch
import torch.nn as nn

# Reference https://zhuanlan.zhihu.com/p/443839799

class GoogLeNet(nn.Module):
    def __init__(self,out_dim):
        super(GoogLeNet,self).__init__()
        self.pre_layer = nn.Sequential(nn.Conv2d(3,192,kernel_size=3,padding=1),
                                       nn.BatchNorm2d(192),nn.ReLU(True)
                                       )
        self.a3 = Inception(192,64,96,128,16,32,32)
        self.b3 = Inception(256,128,128,192,32,96,64)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.a4 = Inception(480,192,96,208,16,48,64)
        self.b4 = Inception(512,160,112,224,24,64,64)
        self.c4 = Inception(512,128,128,256,24,64,64)
        self.d4 = Inception(512,112,144,288,32,64,64)
        self.e4 = Inception(528,256,160,320,32,128,128)
        self.a5 = Inception(832,256,160,320,32,128,128)
        self.b5 = Inception(832,384,192,384,48,128,128)
        self.avgpool = nn.AvgPool2d(kernel_size=16,stride=1)
        self.linear = nn.Linear(1024,out_dim)

    def forward(self,x):
        x = self.pre_layer(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        return x

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception,self).__init__()
        #1*1卷积
        self.branch1 = nn.Sequential(nn.Conv2d(in_planes,n1x1,kernel_size=1),
                                     nn.BatchNorm2d(n1x1),nn.ReLU(True)
                                     )
        #1*1和3*3卷积
        self.branch2 = nn.Sequential(nn.Conv2d(in_planes,n3x3red,kernel_size=1),
                                  nn.BatchNorm2d(n3x3red),nn.ReLU(True),
                                  nn.Conv2d(n3x3red,n3x3,kernel_size=3,padding=1),
                                  nn.BatchNorm2d(n3x3),nn.ReLU(True)
                                  )

        #1*1和2个3*3，相当于是1*1和1个5*5
        self.branch3 = nn.Sequential(nn.Conv2d(in_planes,n5x5red,kernel_size=1),
                                   nn.BatchNorm2d(n5x5red),nn.ReLU(True),
                                   nn.Conv2d(n5x5red,n5x5,kernel_size=3,padding=1),
                                   nn.BatchNorm2d(n5x5),nn.ReLU(True),
                                   nn.Conv2d(n5x5,n5x5,kernel_size=3,padding=1),
                                   nn.BatchNorm2d(n5x5),nn.ReLU(True)
        )

        #3*3池化和1*1
        self.branch4 = nn.Sequential(nn.MaxPool2d(3,stride=1,padding=1),
                                   nn.Conv2d(in_planes,pool_planes,kernel_size=1),
                                   nn.BatchNorm2d(pool_planes),nn.ReLU(True)
                                   )

    def forward(self,x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1,out2,out3,out4],dim=1)