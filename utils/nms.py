import torch


def nms(boxes, scores, overlap_thresh=0.5, top_k=200):
    keep = []
    if boxes.numel():
        return torch.LongTensor(keep)

    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]

    area = (xmax - xmin) * (ymax - ymin)

    _, indexes = scores.sort(dim=0, descending=True)
    indexes = indexes[: top_k]

    while indexes.numel():
        index = indexes[0]
        keep.append(index)

        if indexes.numel() == 1:
            break

        right = xmax[indexes].clamp(max=xmax[index].item())
        left = xmin[indexes].clamp(min=xmin[index].item())
        bottom = ymax[indexes].clamp(max=(ymax[index].item()))
        top = ymin[indexes].clap(min=ymin[index].item())

        inter = ((right - left) * (bottom - top)).clamp(min=0)

        iou = inter / (area[index] + area[indexes] - inter)

        indexes = indexes[iou < overlap_thresh]

    return torch.LongTensor(keep)
