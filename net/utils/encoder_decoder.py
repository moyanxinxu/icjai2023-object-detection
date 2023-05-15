import torch


def center_to_corner(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=1)


def corner_to_center(boxes):
    return torch.cat(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]), dim=1)


def encode(prior, matched_bbox, variance):
    matched_bbox = corner_to_center(matched_bbox)
    output_center = (matched_bbox[:, :2] - prior[:, :2]) / (variance[0] * prior[:, 2:])
    output_width_height = torch.log(matched_bbox[:, 2:] / prior[:, 2:] + 1e-5) / variance[1]
    return torch.cat((output_center, output_width_height), dim=1)


def decode(loc, prior, variance):
    bbox = torch.cat((
        prior[:, :2] + prior[:, 2:] * variance[0] * loc[:, :2],
        prior[:, 2:] * torch.exp(variance[1] * loc[:, 2:])), dim=1)
    bbox = center_to_corner(bbox)
    return bbox
