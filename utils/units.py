import torch
from torch import nn

import torch.nn.functional as F




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

class att(nn.Module):
    def __init__(self, channels,  factor=16):#factorc 32

        super(att, self).__init__()
        self.conv = ConvBNR(48, 16, 3)
        self.groups = factor
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)


    def forward(self, x):
        x=self.conv(x)
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w

        x1 = self.gn(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))

        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights =  torch.matmul(x11, x22).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
class ConvBNR2(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0,dilation=1, bias=False):
        super(ConvBNR2, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            ConvBNR2(inplanes, 32, 1),
            ConvBNR2(32, 32, kernel_size,padding=1,stride=stride,dilation=dilation,bias=bias),
            ConvBNR2(32, planes, 1)
        )


    def forward(self, x):

        x2=self.block(x)

        if x2.size()[2:] != x.size()[2:]:
            x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=False)

        x=x+x2
        return x



class BGM(nn.Module):
    def __init__(self):
        super(BGM, self).__init__()
        self.reduce1 = Conv1x1(256, 64)
        self.reduce2 = Conv1x1(512, 64)
        self.reduce3 = Conv1x1(1024, 64)
        self.reduce4 = Conv1x1(2048, 64)
        self.block00=nn.Sequential(
            ConvBNR(256, 64, 3),

        )

        self.block0=nn.Sequential(
            ConvBlock(64,64)

        )
        self.block1=nn.Sequential(

            ConvBlock(64, 64,stride=2),
        )
        self.block2=nn.Sequential(

            ConvBlock(64, 64,stride=2),
            ConvBlock(64, 64,stride=2),
        )
        self.block3 = nn.Sequential(

            ConvBlock(64, 64,stride=2),
            ConvBlock(64, 64,stride=2),
            ConvBlock(64, 64,stride=2),
        )
        self.block4=nn.Sequential(

            # ConvBNR2(64, 64, 1),
            # ConvBNR2(64, 64, 3, padding=1, stride=1),
            # ConvBNR2(64, 1, 1)
            ConvBlock(64, 64, stride=1),
            nn.Conv2d(64, 1, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


    def forward(self, x4,x3,x2, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)
        x3 = self.reduce3(x3)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size, mode='bilinear', align_corners=False)

        out = torch.concat((x4,x3,x2, x1), dim=1)
        out=self.block00(out)
        out0=self.block0(out)

        out1=self.block1(out)

        out2=self.block2(out)
        out3=self.block3(out)
        out1=F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=False)
        out2=F.interpolate(out2, scale_factor=4, mode='bilinear', align_corners=False)
        out3=F.interpolate(out3, scale_factor=8, mode='bilinear', align_corners=False)

        # out=torch.concat((out0,out1, out2, out3), 1)

        out=torch.add(torch.add(torch.add(out0,out1),out2),out3)
        oout = self.block4(out)


        return oout


class GCAM(nn.Module):
    def __init__(self, hchannel, channel):
        super(GCAM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)

        self.conv1_2 = Conv1x1(channel, channel) #ConvBlock(channel, channel)

        self.conv3_3 = ConvBNR(channel, channel, 3)

        self.block1=ConvBlock(channel//4,channel//4)


    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])

        x1 = self.dconv5_1(xc[1] + x0 + xc[2])

        x2 = self.dconv7_1(xc[2] + x1 + xc[3])

        x3 = self.dconv9_1(xc[3] + x2)
        x33 = self.block1(x3)
        x22 = self.block1(x33+x2)
        x11 = self.block1(x1+x22)
        x00 = self.block1(x0+x11)
        xx = self.conv1_2(torch.cat((x00, x11, x22, x33), dim=1))
        x = self.conv3_3(x + xx)

        return x

class BGM2(nn.Module):
    def __init__(self):
        super(BGM2, self).__init__()
        self.reduce1 = Conv1x1(64, 64)
        self.reduce2 = Conv1x1(128, 64)
        self.reduce3 = Conv1x1(320, 64)
        self.reduce4 = Conv1x1(512, 64)
        self.block00=nn.Sequential(
            ConvBNR(256, 64, 3),

        )

        self.block0=nn.Sequential(
            ConvBlock(64,64)

        )
        self.block1=nn.Sequential(

            ConvBlock(64, 64,stride=2),
        )
        self.block2=nn.Sequential(

            ConvBlock(64, 64,stride=2),
            ConvBlock(64, 64,stride=2),
        )
        self.block3 = nn.Sequential(

            ConvBlock(64, 64,stride=2),
            ConvBlock(64, 64,stride=2),
            ConvBlock(64, 64,stride=2),
        )
        self.block4=nn.Sequential(
            # ConvBNR(256, 1, 3)
            ConvBNR2(64, 64, 1),
            ConvBNR2(64, 64, 3, padding=1, stride=1),
            ConvBNR2(64, 1, 1)
        )

    def forward(self, x4, x3, x2, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)
        x3 = self.reduce3(x3)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size, mode='bilinear', align_corners=False)

        out = torch.concat((x4, x3, x2, x1), dim=1)
        out = self.block00(out)
        out0 = self.block0(out)

        out1 = self.block1(out)

        out2 = self.block2(out)
        out3 = self.block3(out)
        out1 = F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, scale_factor=4, mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, scale_factor=8, mode='bilinear', align_corners=False)

        # out=torch.concat((out0,out1, out2, out3), 1)

        out = torch.add(torch.add(torch.add(out0, out1), out2), out3)
        oout = self.block4(out)

        return oout



if __name__=="__main__":
    moudel=att(32)
    out=moudel(torch.randn(16,32,224,224))
    print(out.shape)