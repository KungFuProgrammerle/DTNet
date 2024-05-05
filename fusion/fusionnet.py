import torch
import torch.nn as nn
import numpy as np
import os
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from  utils.units import  att,ConvBlock
import torch.nn.functional as F


def up(x, scale):
    return F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.block(x)
class MFM(nn.Module):
    def __init__(self ):
        super(MFM, self).__init__()
        self.att=nn.Sequential(

            att(16)
        )

        self.block0 = nn.Sequential(
            ConvBNR(3, 64, 3),
            ConvBNR(64, 64, 3),

        )

        self.block1 = nn.Sequential(
            ConvBlock(64, 64,stride=2),
        )
        self.block2 = nn.Sequential(
            ConvBlock(64, 64,stride=2),
            ConvBlock(64, 64,stride=2),
        )
        self.block3 = nn.Sequential(
            ConvBlock(64, 64, stride=2),
            ConvBlock(64, 64, stride=2),
            ConvBlock(64, 64, stride=2),
        )
        self.block4 = nn.Sequential(
            ConvBlock(64, 64, stride=1),
        )
        self.block5=nn.Sequential(
            # ConvBNR(128, 64, 3),
            nn.Conv2d(128, 1, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=self.block0(x)

        x1=self.block1(x)
        x2=self.block2(x)
        x3=self.block3(x)

        out1 = torch.chunk(x1, 4, dim=1)
        out2 = torch.chunk(x2, 4, dim=1)
        out3 = torch.chunk(x3, 4, dim=1)
        oout1=torch.concat((out1[0],up(out2[0],2),up(out3[0],4)),1)
        # oout1=out1[0]+up(out2[0],2)+up(out3[0],4)
        oout1=self.att(oout1)
        oout2=torch.concat((out1[1],up(out2[1],2),up(out3[1],4)),1)
        # oout2 = out1[1] + up(out2[1], 2) + up(out3[1], 4)
        oout2 = self.att(oout2)

        oout3=torch.concat((out1[2],up(out2[2],2),up(out3[2],4)),1)
        # oout3 = out1[2] + up(out2[2], 2) + up(out3[2], 4)
        oout3 = self.att(oout3)

        oout4=torch.concat((out1[3],up(out2[3],2),up(out3[3],4)),1)
        # oout4 = out1[3] + up(out2[3], 2) + up(out3[3], 4)
        oout4 = self.att(oout4)

        out=torch.concat((oout1,oout2,oout3,oout4),1)

        out=self.block4(out)

        out=torch.concat((x,up(out,2)),1)

        out=self.block5(out)

        return out

if __name__ == '__main__':
    x       = torch.randn(16,3,512,512)
    model   = MFM()
    out=model(x)