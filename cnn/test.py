import argparse
import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from .net.cnn import Net
from .utils.dataloader import test_dataset
from tqdm import  tqdm
from .eval import eval
def test(pth_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=544, help='testing size')
    parser.add_argument('--pth_path', type=str, default=pth_path)
    rootpath= os.path.dirname(os.path.abspath(__file__))
    root= os.path.join(os.path.dirname(rootpath),'Dataset/TestDataset')
    for _data_name in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:
        data_path = root+'/{}/'.format(_data_name)
        save_path = rootpath+'/results*/{}/'.format(_data_name)
        opt = parser.parse_args()
        model = Net()
        model.load_state_dict(torch.load(opt.pth_path))
        model.cuda()
        model.eval()

        # os.makedirs(save_path, exist_ok=True)

        image_root = '{}/Imgs/'.format(data_path)
        gt_root = '{}/GT/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, opt.testsize)

        for i in tqdm(range(test_loader.size)):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res3, res2, res1, _ = model(image) #_, _, res, _ = model(image)
            reslist=[]
            reslist.append(res1)
            reslist.append(res2)
            reslist.append(res3)
            for i,res in enumerate(reslist):
                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                imageio.imwrite(save_path.replace('*', str(i+1)) + name, (res * 255).astype(np.uint8))

   # eval()
if "__main__"==__name__:
    path= '/home/jk-515-4/lgw/TEnet3/cnn/checkpoints/45-276cnn_epoch_best.pth'
    test(path)