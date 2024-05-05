import torch
import torch.nn.functional as F
import numpy as np
from utils.dataloader import get_loader
from tqdm import  tqdm
import os
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from .confusiontest import  test,MFM
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.benchmark = True
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
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
def adjust_lr(optimizer, init_lr, epoch,  total_epoch=30):

    lr = init_lr * (1 - float(epoch) / total_epoch) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def main():

    trainsize = 544 #544
    epochs=100 #31
    lr = 0.01
    model = MFM()
    model.cuda()
    rootpath= os.path.dirname(os.path.abspath(__file__))
    gt_root= os.path.join(os.path.dirname(rootpath),'Dataset/TrainDataset/GT/')
    save_path = os.path.join(rootpath,'checkpoints/')
    train_loader = get_loader(gt_root ,batchsize=16, trainsize=trainsize)
    params = model.parameters()

    optimizer = torch.optim.AdamW(params, lr)
    print('model paramters', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(optimizer)
    model.train()

    for epoch in range(epochs)  :
        adjust_lr(optimizer, lr, epoch, epochs)
        for i, pack in enumerate(tqdm(train_loader)) :
            gt=pack
            gt = gt.cuda()

            x=torch.concat((gt,gt,gt),1).cuda()
            out=model(x)
            loss=dice_loss(out,gt)
            #wbce = F.binary_cross_entropy_with_logits(out, gt, reduce='none')
            loss = loss #+ wbce
            # print(loss.data)
            loss.backward()
            clip_gradient(optimizer, 0.5)
            optimizer.step()
            optimizer.zero_grad()

        if epoch >0:
            finalpath=save_path + 'fusionnet{}img{}.pth'.format(epoch,trainsize)
            torch.save(model.state_dict(), finalpath)
            print('[Saving Snapshot:]', finalpath)
            test(finalpath)

if "__main__"==__name__:
    main()




