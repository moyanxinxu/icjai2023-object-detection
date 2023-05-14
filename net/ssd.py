import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.l2norm import L2Norm
from utils.priorbox import PriorBox
from vgg import vgg16


class SSD(nn.Module):

    def __init__(self, n_classes: int, variance=(0.1, 0.2), top_k=200, conf_thresh=0.01, nms_thresh=0.45,
                 image_size=300, **config):
        super().__init__()
        self.n_classes = n_classes
        self.image_size = image_size
        config['image_size'] = image_size

        # 生成先验框
        self.priorbox_generator = PriorBox(**config)
        self.prior = Tensor(self.priorbox_generator())

        self.vgg = vgg16()
        self.l2norm = L2Norm(512, 20)
        self.extras = nn.ModuleList([
            nn.Conv2d(1024, 256, 1),  # conv8_2
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.Conv2d(512, 128, 1),  # conv9_2
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.Conv2d(256, 128, 1),  # conv10_2
            nn.Conv2d(128, 256, 3),
            nn.Conv2d(256, 128, 1),  # conv11_2
            nn.Conv2d(128, 256, 3),
        ])

        self.confs = nn.ModuleList([
            nn.Conv2d(512, n_classes * 4, 3, padding=1),
            nn.Conv2d(1024, n_classes * 6, 3, padding=1),
            nn.Conv2d(512, n_classes * 6, 3, padding=1),
            nn.Conv2d(256, n_classes * 6, 3, padding=1),
            nn.Conv2d(256, n_classes * 4, 3, padding=1),
            nn.Conv2d(256, n_classes * 4, 3, padding=1),
        ])

        self.locs = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, 3, padding=1),
            nn.Conv2d(1024, 4 * 6, 3, padding=1),
            nn.Conv2d(512, 4 * 6, 3, padding=1),
            nn.Conv2d(256, 4 * 6, 3, padding=1),
            nn.Conv2d(256, 4 * 4, 3, padding=1),
            nn.Conv2d(256, 4 * 4, 3, padding=1),
        ])

    def forward(self, x):
        loc = []
        conf = []
        sources = []

        batch_size = x.size(0)

        for layer in self.vgg[:23]:
            x = layer(x)

        sources.append(self.l2norm(x))

        for layer in self.vgg[23:]:
            x = layer(x)

        sources.append(x)

        for i, layer in enumerate(self.extras):
            x = F.relu(layer(x), inplace=True)
            if i % 2 == 1:
                sources.append(x)

        for x, conf_layer, loc_layer in zip(sources, self.confs, self.locs):
            loc.append(loc_layer(x).permute(0, 2, 3, 1).contiguous())
            conf.append(conf_layer(x).permute(0, 2, 3, 1).contiguous())

        conf = torch.cat([i.view(batch_size, -1) for i in conf], dim=1)
        loc = torch.cat([i.view(batch_size, -1) for i in loc], dim=1)

        return loc.view(batch_size, -1, 4), conf.view(batch_size, -1, self.n_classes), self.prior

    @torch.no_grad()
    def predict(self, x):
        loc, conf, prior = self(x)
        return self.detector(loc, F.softmax(conf, dim=-1), prior.to(loc.device))
