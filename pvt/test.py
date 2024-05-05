import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from tqdm import tqdm
import cv2
from .net.pvt import Net
from .utils.dataloader import My_test_dataset


def test(netPath):
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=704, help='testing size default 352')

    opt = parser.parse_args()

    rootpath = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(os.path.dirname(rootpath), 'Dataset/TestDataset')

    for _data_name in ['CAMO', 'COD10K', 'CHAMELEON', 'NC4K']:
        data_path = root + '/{}/'.format(_data_name)
        save_path = rootpath + '/res*/{}/'.format(_data_name)
        model = Net()
        model.load_state_dict(torch.load(netPath))
        model.cuda()
        model.eval()
        image_root = '{}/Imgs/'.format(data_path)
        gt_root = '{}/GT/'.format(data_path)
        test_loader = My_test_dataset(image_root, gt_root, opt.testsize)
        for i in tqdm(range(test_loader.size)):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res3, res2, res1,oe = model(image) #P3, ,  = model(image)
            reslist=[]
            reslist.append(res1)
            reslist.append(res2)
            reslist.append(res3)
            for i,res in enumerate(reslist):
                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                cv2.imwrite(save_path.replace('*', str(i+1)) + name, res * 255)
if "__main__"==__name__:
    path= '/home/jiao/lgw/bestepoch/fusionnet36img512channelsize0.pth'
    test()