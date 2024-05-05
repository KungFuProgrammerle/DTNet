
from cnn.train import main as cnn
from pvt.Train import main as pvt
from fusion.confusion import main as fusion









if __name__=='__main__':
    print("start----------------cnn")
    cnn()
    print("start----------------pvt") #47886181
                                      #26722149
    pvt()

    print("start----------------fusion")  #141924
    fusion()
    print("finished----------------train and test")