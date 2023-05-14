import torch
from encoder import center_to_corner, encode


def match(overlap_thresh, prior, bbox, variance, label):
    iou = jaccard_overlap(center_to_corner(prior), bbox)

    best_prior_iou, best_prior_index = iou.max(dim=0)

    best_bbox_iou, best_bbox_index = iou.max(dim=1)

    best_bbox_iou.index_fill_(0, best_prior_index, 2)
    for i in range(len(best_prior_index)):
        best_bbox_index[best_prior_index[i]] = i

    matched_bbox = bbox[best_bbox_index]

    conf = label[best_bbox_index] + 1
    conf[best_bbox_iou < overlap_thresh] = 0

    loc = encode(prior, matched_bbox, variance)

    return loc, conf


def jaccard_overlap(prior, bbox):
    A = prior.size(0)
    B = bbox.size(0)
    xy_max = torch.min(prior[:, 2:].unsqueeze(1).expand(A, B, 2),
                       bbox[:, 2:].broadcast_to(A, B, 2))
    xy_min = torch.max(prior[:, :2].unsqueeze(1).expand(A, B, 2),
                       bbox[:, :2].broadcast_to(A, B, 2))

    inter = (xy_max - xy_min).clamp(min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]

    area_prior = ((prior[:, 2] - prior[:, 0]) * (prior[:, 3] - prior[:, 1])).unsqueeze(1).expand(A, B)
    area_bbox = ((bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])).broadcast_to(A, B)

    return inter / (area_prior + area_bbox - inter)
