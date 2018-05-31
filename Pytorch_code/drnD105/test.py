import argparse
import os
import shutil
import time
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os.path as osp
import torch.nn.functional as F
import numpy as np
from augmentation import HorizontalFlip
import drn
from functools import partial

from memcached_dataset import McDataset
from distributed_utils import dist_init, average_gradients, DistModule

from base_tester import BaseTester, TenCropTester

NB_CLASSES = 128
class FinetunePretrainedModels(nn.Module):
    def __init__(self, num_classes, net_cls, net_kwards):
        super().__init__()
        self.net = net_cls(**net_kwards)
        self.net.fc = nn.Conv2d(
            self.net.fc.in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        return self.net(x)
model_dict = {
    'drn_d_105': partial(FinetunePretrainedModels, NB_CLASSES, drn.drn_d_105)
}

net_kwards = [{'pretrained': 'True'}, {'pretrained': 'False'}]

def get_model(model_name: str, pretrained=True):
    # print('[+] getting model architecture... ')
    if(pretrained):
        model = model_dict[model_name](net_kwards[0])
    else:
        model = model_dict[model_name](net_kwards[1])
    print('[+] done.')
    return model

test_root = "/mnt/lustre/yangkunlin/furniture/data/test/"
test_source = "/mnt/lustre/yangkunlin/furniture/data/test0.txt"
image_size = 256
input_size = 224
batch_size = 16
arch = "drnD105"
ckp_path = "/mnt/lustre/yangkunlin/furniture/pytorch/drnD105/checkpoint1_best.pth.tar"

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    normalize
])

preprocess_hflip = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(input_size),
    HorizontalFlip(),
    transforms.ToTensor(),
    normalize
])


def temp_trans(torch_image):
    torch_image = transforms.ToTensor()(torch_image)
    return normalize(torch_image)

preprocess_tencrop = transforms.Compose([
    transforms.Resize(image_size),
    transforms.TenCrop(input_size),
    transforms.Lambda(lambda crops: torch.stack([temp_trans(crop)for crop in crops])),
    # transforms.ToTensor(),
    # normalize
])

def main():

    # tta_preprocess = [preprocess, preprocess_hflip]
    tta_preprocess = [preprocess_tencrop]
    id = 0
    print("testing {}.....".format(ckp_path))

    for trans in tta_preprocess:
        print("id is: {}".format(id))
        test_dataset = McDataset(
            test_root,
            test_source,
            transform=trans)

        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=False)
        print("test loading....")
        model = get_model('drn_d_105', pretrained=False)
        # model.cuda()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = load_checkpoint(ckp_path)
        model.load_state_dict(checkpoint['state_dict'])
        # tester = BaseTester(model)
        tester = TenCropTester(model)
        pred = tester.extract(test_loader)
        np.save("./rst/drnD105_ck1{}.npy".format(id), pred)
        id += 1

if __name__ == '__main__':
    main()
