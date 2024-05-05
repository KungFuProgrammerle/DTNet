import os
import torch
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from datetime import datetime
from .net.cnn import Net
from .utils.dataloader import get_loader
from .utils.utils import clip_gradient, AvgMeter, poly_lr
import torch.nn.functional as F
import numpy as np
from .utils.dataloader import test_dataset
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.benchmark = True
from .test import test

best_mae = 1
best_epoch = 0
def val(model, epoch, save_path,opt):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0

        test_loader = test_dataset(image_root=opt.test_path + '/COD10K/Imgs/',
                            gt_root=opt.test_path + '/COD10K/GT/',
                            testsize=opt.trainsize)

        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res3,res2, res1,_ = model(image)
            # eval Dice
            res = F.upsample(res1, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size

        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'cnn_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def train(train_loader, model, optimizer, epoch,opt,total_step):
    model.train()

    loss_record3, loss_record2, loss_record1, loss_recorde = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):

        # ---- data prepare ----
        images, gts, edges = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        edges = Variable(edges).cuda()
        # ---- forward ----
        lateral_map_3, lateral_map_2, lateral_map_1, edge_map = model(images)

        # ---- loss function ----
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)
        loss1 = structure_loss(lateral_map_1, gts)
        losse = dice_loss(edge_map, edges)
        loss = loss3 + loss2 + loss1 + losse
        # ---- recording loss ----
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record1.update(loss1.data, opt.batchsize)
        loss_recorde.update(losse.data, opt.batchsize)


        # ---- backward ----

        loss.backward()
        clip_gradient(optimizer, opt.clip)

        optimizer.step()
        optimizer.zero_grad()

        # ---- train visualization ----
        if i % 30 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde.avg))



def main():

    rootpath= os.path.dirname(os.path.abspath(__file__))
    gt_root= os.path.join(os.path.dirname(rootpath),'Dataset/TrainDataset')
    te_root = os.path.join(os.path.dirname(rootpath), 'Dataset/TestDataset')
    save_path = rootpath+'/checkpoints/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=40, help='epoch number')#30
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=544,help='training dataset size')#544
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default=gt_root, help='path to train dataset')
    parser.add_argument('--test_path', type=str,default=te_root,help='path to testing dataset')
    parser.add_argument('--train_save', type=str,
                      )
    opt = parser.parse_args()

    # ---- build models ----
    model = Net().cuda()

    params = model.parameters()
    optimizer = torch.optim.AdamW(params, opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)
    print('model paramters', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(optimizer)
    print("Start Training")
    for epoch in  range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch,opt,total_step)
        if epoch>25:
            val(model, epoch, save_path, opt)

    netPath = rootpath + '/checkpoints/cnn_epoch_best.pth'
    test(netPath)

        


if __name__ == '__main__':
    main()
