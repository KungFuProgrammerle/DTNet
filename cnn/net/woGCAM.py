import torch
import torch.nn as nn
import torch.nn.functional as F






class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()



        self.reduce1=nn.Sequential(
            Conv1x1(256, 64),
            # ConvBNR(128, 64, 3),
            nn.Conv2d(64, 1, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.reduce2 = nn.Sequential(
            Conv1x1(512, 64),
            # ConvBNR(128, 64, 3),
            nn.Conv2d(64, 1, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.reduce3 = nn.Sequential(
            Conv1x1(1024, 64),
            # ConvBNR(128, 64, 3),
            nn.Conv2d(64, 1, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.reduce4 = nn.Sequential(
            Conv1x1(2048, 64),
            # ConvBNR(128, 64, 3),
            nn.Conv2d(64, 1, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )




    def forward(self, x4, x3, x2, x1):
        # B Seq

        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)
        x3 = self.reduce3(x3)
        x4 = self.reduce4(x4)

        x3 = F.interpolate(x3, (544,544), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, (544,544), mode='bilinear', align_corners=False)
        x1 = F.interpolate(x1, (544,544), mode='bilinear', align_corners=False)

        return x3, x2, x1



