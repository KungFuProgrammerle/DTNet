
import torch
import torch.nn.functional as F
import numpy as np
from utils.dataloader import get_loader_test1
from tqdm import  tqdm
import os
from .fusionnet import MFM
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import imageio
from .confunsion_eval import confunsionevl

def test(path):
    model = MFM()
    model.load_state_dict(torch.load(path))
    model.cuda()
    model.eval()
    trainsize=544
    rootpath= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for _data_name in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:


        image_root1 = rootpath + '/cnn/results1/{}/'.format(_data_name)
        image_root2 = rootpath + '/cnn/results2/{}/'.format(_data_name)
        image_root3 = rootpath + '/cnn/results3/{}/'.format(_data_name)
        image_root4 = rootpath + '/pvt/res1/{}/'.format(_data_name)
        image_root5 = rootpath + '/pvt/res2/{}/'.format(_data_name)


        image_root6 = rootpath + '/pvt/res3/{}/'.format(_data_name)

        gt_root=os.path.join(rootpath,'Dataset/TestDataset/{}/GT/'.format(_data_name))

        train_loader = get_loader_test1(image_root1,image_root2, image_root3,image_root4,image_root5, image_root6,gt_root ,batchsize=1, trainsize=trainsize)



        save_path=os.path.join(rootpath,'fusion/fusionresult/{}/'.format(_data_name))
        for i, pack in tqdm(enumerate(train_loader)):
            image1,image2,image3,image4,image5,image6,gt,name=pack
            image1 = image1.cuda()
            image2 = image2.cuda()
            image3 = image3.cuda()
            image4 = image4.cuda()
            image5 = image5.cuda()
            image6 = image6.cuda()
            gt = gt.cuda()
            # res= torch.max(torch.max(torch.max(torch.max(torch.max(image1,image3),image2),image4) ,image5) ,image6)
            res=(image1+image2+image3+image4+image5+image6)/6
            res = F.upsample(res, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            res[res > 0.5] = 255
            # res = res * 255

            imageio.imwrite(save_path + name[0], (res).astype(np.uint8))

    confunsionevl()

if "__main__"==__name__:
    path= '/home/jiao/lgw/fusion/checkpoints/fusionnet36img512.pth'
    test(path)










