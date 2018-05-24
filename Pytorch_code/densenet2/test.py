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

from memcached_dataset import McDataset
from distributed_utils import dist_init, average_gradients, DistModule

from base_tester import BaseTester

class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet') :
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(original_model.fc.in_features, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg16'
        elif arch.startswith('dense'):
            self.features = nn.Sequential(*list(original_model.children())[:-1])

            # Get number of features of last layer
            num_feats = original_model.classifier.in_features

            # Plug our classifier
            self.classifier = nn.Sequential(
                nn.Linear(num_feats, num_classes)
            )
            self.modelName = 'densenet'
        else :
            raise("Finetuning not supported on this architecture yet")

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'densenet':
            f = F.relu(f, inplace=True)
            f = F.avg_pool2d(f, kernel_size=7).view(f.size(0), -1)
        y = self.classifier(f)
        return y

test_root = "/mnt/lustre/yangkunlin/furniture/data/test/"
test_source = "/mnt/lustre/yangkunlin/furniture/data/test0.txt"
image_size = 256
input_size = 224
batch_size = 10
arch = "densenet201"
ckp_path = "/mnt/lustre/yangkunlin/furniture/pytorch/densenet2/checkpoint2_best.pth.tar"

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_dataset = McDataset(
        test_root,
        test_source,
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=False)

    model = models.__dict__[arch]()
    model = FineTuneModel(model, arch, 128)
    # model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = load_checkpoint(ckp_path)
    model.load_state_dict(checkpoint['state_dict'])
    tester = BaseTester(model)
    pred = tester.extract(test_loader)
    np.save("./rst/prob_dense.npy", pred)

if __name__ == '__main__':
    main()
