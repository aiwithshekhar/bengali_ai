from models_resnet import *
import torch.nn as nn
from torch.nn import functional as F


class modi_resnet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        if pretrained is True:
            self.model = resnet34(pretrained="imagenet")
        else:
            self.model = resnet34(pretrained=None)

        self.l0 = nn.Linear(512, 168)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 7)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0, l1, l2
