import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


# define the module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(  # ->(3,32,32)
            nn.Conv2d(
                in_channels=3,  # gray = 1,RGB=3
                out_channels=16,  # filter number
                kernel_size=5,
                stride=1,
                padding=2,  # if stride = 1,padding = (kernel_size-1)/stride = (5-1)/2
            ),
            nn.ReLU(),  # ->(16,32,32)
            nn.MaxPool2d(kernel_size=2),  # ->(16,16,16)
        )

        self.conv2 = nn.Sequential(  # ->(16,16,16)
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),  # ->(32,16,16)
            nn.MaxPool2d(kernel_size=2)  # ->(32,8,8)
        )

        self.conv3 = nn.Sequential(  # ->(32,8,8)
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),  # ->(64,8,8)
            nn.MaxPool2d(kernel_size=2)  # ->(64,4,4)
        )
        self.out1 = nn.Linear(64 * 4 * 4, 526)  # 将数据展成一列
        self.out2 = nn.Linear(526, 10)  # 将数据展成一列

    def forward(self, x):
        x = self.conv1(x)  # type is tuple
        x = self.conv2(x)  # type is tensor [batch,32,7,7]
        x = self.conv3(x)  # type is tensor [batch,32,7,7]
        x = x.view(x.size(0), -1)  # -1：展成一列 [batch,32*7*7]
        x = self.out1(x)
        output = self.out2(x)
        return output


