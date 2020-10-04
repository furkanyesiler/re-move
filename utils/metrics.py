import os

import torch


def average_precision(ypred, path='', k=None, eps=1e-10, reduce_mean=True, dataset=0, print_metrics=True):
    """
    calculating performance metrics
    :param ypred: square distance matrix
    :param path: the path of the working directory
    :param k: k value for map@k
    :param eps: epsilon value for numerical stability
    :param reduce_mean: whether to take mean of the average precision values of each query
    :param dataset: which dataset to evaluate (required for loading the ground truth)
    :param print_metrics: whether to print metrics
    :return: mean average precision value
    """
    if dataset == 0:  # loading the ground truth for our validation set
        ytrue = os.path.join(path, 'data/ytrue_val.pt')
        ytrue = torch.load(ytrue).float()
    elif dataset == 1:  # loading the ground truth for Da-TACOS
        ytrue = os.path.join(path, 'data/ytrue_benchmark.pt')
        ytrue = torch.load(ytrue).float()

    if k is None:
        k = ypred.size(1)
    _, spred = torch.topk(ypred, k, dim=1)
    found = torch.gather(ytrue, 1, spred)

    temp = torch.arange(k).float() * 1e-6
    _, sel = torch.topk(found - temp, 1, dim=1)
    mrr = torch.mean(1/(sel+1).float())
    mr = torch.mean((sel+1).float())
    top1 = torch.sum(found[:, 0])
    top10 = torch.sum(found[:, :10])

    pos = torch.arange(1, spred.size(1)+1).unsqueeze(0).to(ypred.device)
    prec = torch.cumsum(found, 1)/pos.float()
    mask = (found > 0).float()
    ap = torch.sum(prec*mask, 1)/(torch.sum(ytrue, 1)+eps)
    ap = ap[torch.sum(ytrue, 1) > 0]
    if print_metrics:
        print('mAP: {:.3f}'.format(ap.mean().item()))
        print('MRR: {:.3f}'.format(mrr.item()))
        print('MR: {:.3f}'.format(mr.item()))
        print('Top1: {:.0f}'.format(top1.item()))
        print('Top10: {:.0f}'.format(top10.item()))
    return ap.mean() if reduce_mean else ap


def pairwise_euclidean_distance(x, y=None, eps=1e-12):
    """
    computing squared Euclidean distances between the elements of two tensors
    :param x: first tensor
    :param y: second tensor (optional)
    :param eps: epsilon value for avoiding div by zero
    :return: pairwise distance matrix
    """
    x_norm = x.pow(2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = y.pow(2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2 * torch.mm(x, y.t().contiguous())
    return dist + eps


def pairwise_cosine_similarity(x, y=None):
    """
    computing cosine similarity between the elements of two tensors
    :param x: first tensor
    :param y: second tensor (optional)
    :return: pairwise similarity matrix
    """
    if y is None:
        y = x
    return (x @ y.t()).div(y.norm(dim=1)).div(x.norm(dim=1).unsqueeze(-1))


def pairwise_pearson_coef(x, y=None):
    """
    computing Pearson correlation coefficient values between the elements of two tensors
    :param x: first tensor
    :param y: second tensor (optional)
    :return: pairwise similarity matrix
    """
    if y is None:
        y = x

    covar = (x - torch.mean(x, dim=1).view(-1, 1)) @ (y - torch.mean(y, dim=1).view(-1, 1)).t() / (x.size(1) - 1)
    var1 = torch.var(x, dim=1).view(-1, 1)
    var2 = torch.var(y, dim=1).view(1, -1)

    return covar / torch.sqrt(var1 * var2)
