import torch
import torch.nn as nn

from utils.metrics import pairwise_pearson_coef


def group_loss(data, labels, proxies, **kwargs):
    """
    Group loss function. It uses Pearson correlation coefficient as the similarity metric.
    The original paper uses a linear layer instead of proxies but we can think of the proxies
    as the rows of the linear layer.
    :param data: embeddings of each item in the batch
    :param labels: labels of each item in the batch
    :param proxies: proxy vectors for each class
    :param kwargs: any other arguments
    :return: average loss value for the batch
    """
    # separate anchors and others (1 anchor per label)
    all_idx = torch.arange(0, data.size(0))
    anchor_idx = torch.arange(0, data.size(0), 4)
    all_idx[anchor_idx] = 0
    other_idx = all_idx.nonzero().flatten()

    # restructure the batch so that the anchors are at the bottom
    embs = torch.cat((data[other_idx], data[anchor_idx]))
    labels = torch.cat((labels[other_idx], labels[anchor_idx]))

    # compute pairwise similarities
    sim_all = pairwise_pearson_coef(embs)

    # remove negative values with a ReLU function
    sim_all = nn.ReLU()(sim_all)

    # replace similarities of anchors with 0
    sim_all[-anchor_idx.size(0):, -anchor_idx.size(0):] = 0

    # create an inverted identity matrix and send it to cuda if needed
    diag_zero = (1 - torch.eye(sim_all.size(0)))
    if torch.cuda.is_available():
        diag_zero = diag_zero.cuda()

    # mask the diagonal with zeros
    sim_all = sim_all * diag_zero

    # normalize the proxies (or rows of the linear layer)
    proxies = nn.functional.normalize(proxies, p=2, dim=1)

    # compute likelihoods between proxies and non-anchors
    priors = torch.softmax(torch.matmul(embs[:-anchor_idx.size(0)], proxies.t()), dim=1)

    # create a vector for holding proxy labels and send it to cuda if needed
    label_array = torch.arange(proxies.size(0)).unsqueeze(-1).float()
    if torch.cuda.is_available():
        label_array = label_array.cuda()

    # create a mask for positives and send it to cuda if needed
    mask_pos = torch.zeros(labels.size(0), label_array.size(0))
    mask_pos.scatter_(1, labels.long().cpu(), 1)
    if torch.cuda.is_available():
        mask_pos = mask_pos.cuda()

    # appending likelihoods between anchors and their true proxies as 1
    priors = torch.cat((priors, mask_pos[-anchor_idx.size(0):].float()), dim=0)

    # support matrix
    support = torch.matmul(sim_all, priors)

    # refining likelihoods in an iterative way
    for _ in range(3):
        priors = priors * support / (torch.sum(priors * support, dim=1, keepdim=True) + 1e-8)

    # computing the final likelihoods and returning the loss for non-anchors
    r = -1 * torch.log(torch.gather(priors, 1, labels.long()) + 1e-8)
    return torch.mean(r[:-anchor_idx.size(0)])
