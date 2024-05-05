
from cnn.test import test as cnntest
from pvt.test import test as pvttest
from fusion.confusiontest import test as fusiontest






if __name__=='__main__':

    print("start----------------test")
    # cnntest('/home/jk-515-4/lgw/TEnet/bestEpoch/40-0.0273cnn_epoch_best.pth')
    # # #
    # pvttest('/home/jk-515-4/lgw/TEnet/bestEpoch/55-0.0211-1pvt_epoch_best.pth')

    fusiontest('D:\pythonpoject\DTNet\\bestEpoch\\bce+fusionnet8img544.pth')
    print("finished----------------test")