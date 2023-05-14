import torch
from torch import Tensor
from nms import nms
from encoder import decode


class Detector:
    def __init__(self, num_classes, variance, top_k=200, conf_thresh=0.01, nms_thresh=0.45):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.variance = variance
        self.top_k = top_k

    def __call__(self, loc, conf, prior):

        batch_size = loc.size(0)
        out = torch.zeros(batch_size, self.num_classes, self.top_k, 5)

        for i in range(batch_size):
            bbox = decode(loc[i], prior, self.variance)
            conf_score = conf[i].clone()

            for c in range(1, self.num_classes):
                mask = conf_score[:, c] > self.conf_thresh
                scores = conf_score[:, c][mask]

                if scores.size(0) == 0:
                    continue
                boxes = bbox[mask]
                indexes = nms(boxes, scores, self.nms_thresh, self.top_k)
                out[i, c, :len(indexes)] = torch.cat(
                    (boxes[indexes], scores[indexes].unsqueeze(1)), dim=1)

        return out
