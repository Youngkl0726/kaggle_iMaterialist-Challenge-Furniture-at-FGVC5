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
from functools import partial
import dpn

from memcached_dataset import McDataset
from distributed_utils import dist_init, average_gradients, DistModule

from base_tester import BaseTester, TenCropTester


NB_CLASSES = 128
class FinetunePretrainedModels(nn.Module):
    def __init__(self, num_classes, net_cls, net_kwards):
        super().__init__()
        self.net = net_cls(**net_kwards)
        self.net.classifier = nn.Conv2d(
            self.net.classifier.in_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        return self.net(x)

model_dict = {
    'dpn107': partial(FinetunePretrainedModels, NB_CLASSES, dpn.dpn107)
}
net_kwards = [{'pretrained': 'imagenet+5k'}, {'pretrained': None}]

def get_model(model_name: str, pretrained=False):
    # print('[+] getting model architecture... ')
    if(pretrained):
        model = model_dict[model_name](net_kwards[0])
    else:
        model = model_dict[model_name](net_kwards[1])
    print('[+] done.')
    return model

test_root = "/mnt/lustre/yangkunlin/furniture/data/test/"
test_source = "/mnt/lustre/yangkunlin/furniture/data/test0.txt"
# test_root = "/mnt/lustre/yangkunlin/furniture/data/val/"
# test_source = "/mnt/lustre/yangkunlin/furniture/data/valid.txt"
image_size = 256
input_size = 224
batch_size = 16
arch = "dpn107"
ckp_path = "/mnt/lustre/yangkunlin/furniture/pytorch/dpn107/checkpoint5_best.pth.tar"

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


normalize = transforms.Normalize(mean=[124.0 / 255.0, 117.0 / 255.0, 104.0 / 255.0],
                                     std=[1.0 / (.0167 * 255)] * 3)

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

    TTA2_preprocess = [preprocess, preprocess_hflip]
    TTA10_preprocess = [preprocess_tencrop]
    TTA12_preprocess = [preprocess, preprocess_hflip, preprocess_tencrop]
    id = 0
    print("testing {}.....".format(ckp_path))

    for trans in TTA10_preprocess:
        print("id is: {}".format(id))
        test_dataset = McDataset(
            test_root,
            test_source,
            transform=trans)

        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=False)
        print("test loading....")
        model = get_model('dpn107', pretrained=False)
        # model.cuda()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = load_checkpoint(ckp_path)
        model.load_state_dict(checkpoint['state_dict'])
        tester = TenCropTester(model)
        # if id == 2:
        #     tester = TenCropTester(model)
        # else:
        #     tester = BaseTester(model)

        pred = tester.extract(test_loader)
        np.save("./rst/dpn107_ck5{}.npy".format(id), pred)
        id += 1
    # print("test loading....")
    # model = models.__dict__[arch]()
    # model = FineTuneModel(model, arch, 128)
    # # model.cuda()
    # model = torch.nn.DataParallel(model).cuda()
    # checkpoint = load_checkpoint(ckp_path)
    # model.load_state_dict(checkpoint['state_dict'])
    # tester = BaseTester(model)
    # pred = tester.extract(dataloaders)
    # # pred = tester.extract(test_loader)
    # np.save("./rst/prob_dense.npy", pred)

if __name__ == '__main__':
    main()
