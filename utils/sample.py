import torch
import torch.nn.functional as F


@torch.no_grad()
def hard_negative_mining(conf_pred, conf_t, neg_pos_ratio):
    loss = -F.log_softmax(conf_pred, dim=2)[:, :, 0]

    pos_mask = conf_t > 0
    n_pos = pos_mask.long().sum(dim=1, keepdim=True)
    n_neg = n_pos * neg_pos_ratio

    loss[pos_mask] = 0
    _, indexes = loss.sort(dim=1, descending=True)
    _, rank = indexes.sort(dim=1)
    neg_mask = rank < n_neg

    return pos_mask | neg_mask
