
from .pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.units import GCAM,BGM2
from .fsp import Model as FSP
from .feder import SED
from .woGCAM import Model as woGCAM





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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bgm = BGM2()
        self.reduce1 = Conv1x1(64, 64)
        self.reduce2 = Conv1x1(128, 128)
        self.reduce3 = Conv1x1(320, 256)
        self.reduce4 = Conv1x1(512, 256)
        self.predictor1 = nn.Conv2d(64, 1, 1)
        self.predictor2 = nn.Conv2d(128, 1, 1)
        self.predictor3 = nn.Conv2d(256, 1, 1)
        self.gcam1 = GCAM(128, 64)
        self.gcam2 = GCAM(256, 128)
        self.gcam3 = GCAM(256, 256)
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        rootpath = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(os.path.dirname(os.path.dirname(rootpath)), 'pretrain/pvt_v2_b2.pth')
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        # self.FSP=FSP()
        # self.SED=SED(64,64)
        # self.woGCAM=woGCAM()


    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        edge_att = self.bgm(x4, x3, x2, x1)
        # edge_att = self.SED(x4, x3, x2, x1)

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

        # o4, o3, o2, o1=self.FSP(x4, x3, x2, x1)
        # oe=None
        # o3,o2,o1=self.woGCAM(x4, x3, x2, x1)
        return o3,o2,o1,oe


if __name__ == '__main__':
    model = Net().cuda()
    input_tensor = torch.randn(8, 3, 704, 704).cuda()
    prediction1, prediction2,p3,oe = model(input_tensor)
    print(prediction1.size(), prediction2.size(),p3.size())
