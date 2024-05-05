import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random

random.seed(2021)


class CamObjDataset(data.Dataset):
    def __init__(self, gt_root, trainsize):
        self.trainsize = trainsize
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]

        self.gts = sorted(self.gts)



        self.size = len(self.gts)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])



        
    def __getitem__(self, index):

        gt = self.binary_loader(self.gts[index])
        gt = self.img_transform(gt)
        return gt



    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')



    def __len__(self):
        return self.size


class CamObjDataset_test(data.Dataset):
    def __init__(self, image_root1, image_root2, image_root3, gt_root, trainsize):
        self.trainsize = trainsize
        self.images1 = [image_root1 + f for f in os.listdir(image_root1) if f.endswith('.png')]
        self.images2 = [image_root2 + f for f in os.listdir(image_root2) if f.endswith('.png')]
        self.images3 = [image_root3 + f for f in os.listdir(image_root3) if f.endswith('.png')]

        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.names = sorted([f for f in os.listdir(gt_root) if f.endswith('.png')])

        self.images1 = sorted(self.images1)
        self.images2 = sorted(self.images2)
        self.images3 = sorted(self.images3)

        self.gts = sorted(self.gts)

        self.size = len(self.images1)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gt_transform = transforms.Compose([

            transforms.ToTensor()])

    def __getitem__(self, index):
        gt = self.binary_loader(self.gts[index])
        images1 = self.binary_loader(self.images1[index])
        images2 = self.binary_loader(self.images2[index])
        images3 = self.binary_loader(self.images3[index])

        image1 = self.img_transform(images1)
        image2 = self.img_transform(images2)
        image3 = self.img_transform(images3)
        gt=self.gt_transform(gt)

        return image1, image2, image3, gt,self.names[index]

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size






def get_loader(gt_root, batchsize, trainsize, shuffle=True, num_workers=16, pin_memory=True):
    dataset = CamObjDataset(gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader
def get_loader_test(image_root1, image_root2, image_root3,gt_root, batchsize, trainsize, shuffle=False, num_workers=0, pin_memory=True):
    dataset = CamObjDataset_test(image_root1, image_root2, image_root3,gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader
class CamObjDataset_test1(data.Dataset):
    def __init__(self, image_root1, image_root2, image_root3, image_root4, image_root5,image_root6,gt_root, trainsize):
        self.trainsize = trainsize
        self.images1 = [image_root1 + f for f in os.listdir(image_root1) if f.endswith('.png')]
        self.images2 = [image_root2 + f for f in os.listdir(image_root2) if f.endswith('.png')]
        self.images3 = [image_root3 + f for f in os.listdir(image_root3) if f.endswith('.png')]
        self.images4 = [image_root4 + f for f in os.listdir(image_root4) if f.endswith('.png')]
        self.images5 = [image_root4 + f for f in os.listdir(image_root5) if f.endswith('.png')]
        self.images6 = [image_root4 + f for f in os.listdir(image_root6) if f.endswith('.png')]

        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.names = sorted([f for f in os.listdir(gt_root) if f.endswith('.png')])

        self.images1 = sorted(self.images1)
        self.images2 = sorted(self.images2)
        self.images3 = sorted(self.images3)
        self.images4 = sorted(self.images4)
        self.images5 = sorted(self.images5)
        self.images6 = sorted(self.images6)

        self.gts = sorted(self.gts)

        self.size = len(self.images1)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gt_transform = transforms.Compose([

            transforms.ToTensor()])

    def __getitem__(self, index):
        gt = self.binary_loader(self.gts[index])
        images1 = self.binary_loader(self.images1[index])
        images2 = self.binary_loader(self.images2[index])
        images3 = self.binary_loader(self.images3[index])
        images4 = self.binary_loader(self.images4[index])
        images5 = self.binary_loader(self.images5[index])
        images6 = self.binary_loader(self.images6[index])

        image1 = self.img_transform(images1)
        image2 = self.img_transform(images2)
        image3 = self.img_transform(images3)
        image4 = self.img_transform(images4)
        image5 = self.img_transform(images5)
        image6 = self.img_transform(images6)
        gt=self.gt_transform(gt)

        return image1, image2, image3, image4,image5, image6, gt,self.names[index]

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size






def get_loader(gt_root, batchsize, trainsize, shuffle=True, num_workers=16, pin_memory=True):
    dataset = CamObjDataset(gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader
def get_loader_test1(image_root1, image_root2, image_root3,image_root4, image_root5,image_root6,gt_root, batchsize, trainsize, shuffle=False, num_workers=0, pin_memory=True):
    dataset = CamObjDataset_test1(image_root1, image_root2, image_root3,image_root4, image_root5,image_root6,gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader
