import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from net.utils.l2norm import L2Norm
from net.utils.priorbox import PriorBox
from net.vgg import vgg16
from net.detection_head import extra,confs,locs

class SSD(nn.Module):

    def __init__(self, num_classes, variance=(0.1, 0.2), top_k=200, conf_thresh=0.01, nms_thresh=0.45,
                 image_size=300):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        self.priorboxs = PriorBox()
        self.prior = Tensor(self.priorboxs())

        self.vgg = vgg16()
        self.l2norm = L2Norm(512, 20)

        self.extras= extra()       
        self.locs = locs()
        self.confs = confs(num_classes)

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
            # shape : batch_size, 4 * num_anchors, height, width
            # shape : batch_size, height, width, 4 * num_anchors
            loc.append(loc_layer(x).permute(0, 2, 3, 1).contiguous())
            # batch_size, num_classes * num_anchors, height, width
            # batch_size, height, width, num_classes * num_anchors
            conf.append(conf_layer(x).permute(0, 2, 3, 1).contiguous())

        conf = torch.cat([i.view(batch_size, -1) for i in conf], dim=1)
        loc = torch.cat([i.view(batch_size, -1) for i in loc], dim=1)

        return loc.view(batch_size, -1, 4), conf.view(batch_size, -1, self.num_classes), self.prior

    @torch.no_grad()
    def predict(self, x):
        loc, conf, prior = self(x)
        return self.detector(loc, F.softmax(conf, dim=-1), prior.to(loc.device))

if __name__ == '__main__':
    x = torch.rand((1, 3, 300, 300))
    net = SSD(10)
    net(x)
    
    