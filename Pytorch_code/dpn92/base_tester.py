from __future__ import absolute_import

import torch
import torchvision
import numpy as np

class BaseTester():
    def __init__(self, model):
        self.model = model

    def extract(self, data_loader):
        self.model.eval()
        res_features = []

        for batch_index, (data, label) in enumerate(data_loader):
            print("batch_index is: {}".format(batch_index))
            data = data.cuda()
            data = torch.autograd.Variable(data, volatile=True)
            output = self.model(data)
            output = output.data.cpu()
            res_features.extend(output.numpy())
        return np.array(res_features)


class TenCropTester():
    def __init__(self, model):
        self.model = model

    def extract(self, data_loader):
        self.model.eval()
        res_features = []

        for batch_index, (data, label) in enumerate(data_loader):
            print("batch_index is: {}".format(batch_index))
            data = data.cuda()
            data = torch.autograd.Variable(data, volatile=True)
            bs, ncrops, c, h, w = data.size()
            # print("ncrops is:{}".format(ncrops))
            output = self.model(data.view(-1, c, h, w))
            output = output.data.cpu()
            output_avg = output.view(bs, ncrops, -1).mean(1).view(bs,-1)
            res_features.extend(output_avg.numpy())
        return np.array(res_features)