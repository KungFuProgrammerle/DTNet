import torch
from torch.autograd import Variable
import os
import argparse
from .net.pvt import Net
from .utils.dataloader import get_loader, test_dataset
from .utils.utils import clip_gradient, adjust_lr
import torch.nn.functional as F
import numpy as np
from  tqdm import tqdm
from .test import test
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
best_mae = 1
best_epoch = 0
def load_matched_state_dict(model, state_dict, print_stats=True):

    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')

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
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


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

            res3,res2, res1,oe = model(image)
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
                torch.save(model.state_dict(), save_path + 'pvt_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))



def train(train_loader, model, optimizer, epoch, test_path,opt):
    model.train()
    global best
    size_rates = [1]

    for i, pack in tqdm(enumerate(train_loader, start=1)) :
        for rate in size_rates:
            optimizer.zero_grad()

            images, gts,egs = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            egs = Variable(egs).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            lateral_map_3, lateral_map_2, lateral_map_1,oe = model(images)
            # ---- loss function ----
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss1 = structure_loss(lateral_map_1, gts)
            losse = dice_loss(oe, egs)
            loss = loss3 + loss2 + loss1+losse


            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

    # save_path = opt.save_path
    # if epoch % opt.epoch_save == 0:
    #     torch.save(model.state_dict(), save_path + str(epoch) + 'gcem.pth')
def main():



    rootpath= os.path.dirname(os.path.abspath(__file__))
    gt_root= os.path.join(os.path.dirname(rootpath),'Dataset/TrainDataset')
    te_root= os.path.join(os.path.dirname(rootpath),'Dataset/TestDataset')
    check= os.path.join(rootpath,'checkpoints/')

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,default=55, help='epoch number')#50
    parser.add_argument('--lr', type=float,default=1e-4, help='learning rate')#1e-4
    parser.add_argument('--optimizer', type=str,default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation',default=True, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int,default=8, help='training batch size')#8
    parser.add_argument('--trainsize', type=int,default=704, help='training dataset size,candidate=352,704,1056')
    parser.add_argument('--clip', type=float,default=0.5, help='gradient clipping margin')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--decay_rate', type=float,default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,default=gt_root,help='path to train dataset')
    parser.add_argument('--test_path', type=str,default=te_root,help='path to testing dataset')
    parser.add_argument('--save_path', type=str,default=check)
    parser.add_argument('--epoch_save', type=int,default=1, help='every n epochs to save model')
    opt = parser.parse_args()



    model = Net().cuda()


    if opt.load is not None:
        pretrained_dict=torch.load(opt.load)
        print('!!!!!!Sucefully load model from!!!!!! ', opt.load)
        load_matched_state_dict(model, pretrained_dict)


    print('model paramters',sum(p.numel() for p in model.parameters() if p.requires_grad))



    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)
    train_loader = get_loader(image_root, gt_root,edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)




    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch) :
        adjust_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch, opt.save_path,opt)
        if epoch >25:
            val( model, epoch, opt.save_path,opt)
    netPath = rootpath + '/checkpoints/pvt_epoch_best.pth'
    test(netPath)
if __name__ == '__main__':

    main()