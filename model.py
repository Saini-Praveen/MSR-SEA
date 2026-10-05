import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MSRBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSRBlock, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.fuse1 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        self.se = SEBlock(in_channels)
        self.fuse2 = nn.Conv2d(2*out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        conv3_out = F.relu(self.conv3(x))
        conv5_out = F.relu(self.conv5(x))
        concat = torch.cat([conv3_out, conv5_out], dim=1)
        fused = self.fuse1(concat)
        se_out = self.se(fused)

        skip = x + se_out
        fused = self.fuse2(skip)
        return fused

class MSRSEA(nn.Module):
    def __init__(self):
        super(MSRSEA, self).__init__()

        self.init_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.init_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.block1 = MSRBlock(128, 64)
        self.block2 = MSRBlock(64, 32)

        self.final_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.init_conv1(x))   
        x = F.relu(self.init_conv2(x))   

        x = self.block1(x)  
        x = self.block2(x)  

        x = F.relu(self.final_conv1(x))  
        x = self.final_conv2(x)          

        return x
