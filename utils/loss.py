import torch


def bpr_loss(pos_scores, neg_scores):

    loss = -torch.mean(
        torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
    )

    return loss