
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.Res2Net import res2net101_v1b_26w_4s

from utils.units import GCAM,BGM
from .feder import SED
from .fsp import Model as FSP
from .woGCAM import Model as woGCAM

class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)



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




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = res2net101_v1b_26w_4s(pretrained=True)

        self.bgm = BGM()

        self.reduce1 = Conv1x1(256, 64)
        self.reduce2 = Conv1x1(512, 128)
        self.reduce3 = Conv1x1(1024, 256)
        self.reduce4 = Conv1x1(2048, 256)
        self.gcam1 = GCAM(128, 64)
        self.gcam2 = GCAM(256, 128)
        self.gcam3 = GCAM(256, 256)

        self.predictor1 = nn.Conv2d(64, 1, 1)
        self.predictor2 = nn.Conv2d(128, 1, 1)
        self.predictor3 = nn.Conv2d(256, 1, 1)
        # self.sed=SED(64,64)
        # self.FSP=FSP()
        # self.woGCAM = woGCAM()


    def forward(self, x):
        x1, x2, x3, x4 = self.resnet(x)
        edge_att = self.bgm(x4,x3,x2, x1)
        # edge_att=self.sed(x4,x3,x2, x1)


        x1r = self.reduce1(x1)
        x2r = self.reduce2(x2)
        x3r = self.reduce3(x3)
        x4r = self.reduce4(x4)

        x34 = self.gcam3(x3r, x4r)
        x234 = self.gcam2(x2r, x34)
        x1234 = self.gcam1(x1r, x234)

        o3 = self.predictor3(x34)
        o3 = F.interpolate(o3, scale_factor=16, mode='bilinear', align_corners=False)
        o2 = self.predictor2(x234)
        o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=False)
        o1 = self.predictor1(x1234)
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)
        # o4, o3, o2, o1 = self.FSP(x4, x3, x2, x1)
        # oe=None
        # o3,o2,o1=self.woGCAM(x4, x3, x2, x1)

        return o3, o2, o1,oe


if __name__ == '__main__':
    x       = torch.randn(16,3,512,512)
    model   = Net()
    res1,res2,res3,res4=model(x)