import torch

from utils.metrics import pairwise_euclidean_distance


def proxynca_loss(data, labels, emb_size, proxies, **kwargs):
    """
    ProxyNCA loss function. It uses normalized Euclidean distance as explained in the paper.
    :param data: embeddings of each item in the batch
    :param labels: labels of each item in the batch
    :param emb_size: size of the embeddings for normalization
    :param proxies: proxy vectors for each class
    :param kwargs: any other arguments
    :return: average loss value for the batch
    """
    # compute pairwise Euclidean distances
    dist_all = pairwise_euclidean_distance(data, proxies)

    # normalize by the embedding size
    dist_all /= emb_size

    # compute exponential values of the distances
    dist_all = torch.exp(-1 * dist_all)

    # create a vector for holding proxy labels
    label_array = torch.arange(proxies.size(0)).unsqueeze(-1).float()

    # sending the proxy label vector to cuda if needed
    if torch.cuda.is_available():
        label_array = label_array.cuda()

    # creating a mask for the positives for each element
    mask_pos = torch.zeros(labels.size(0), label_array.size(0))
    mask_pos.scatter_(1, labels.long().cpu(), 1)

    # creating a mask for the negatives for each element
    mask_neg = (1 - mask_pos)

    # sending the masks to cuda if needed
    if torch.cuda.is_available():
        mask_pos = mask_pos.cuda()
        mask_neg = mask_neg.cuda()

    # compute the ProxyNCA loss
    temp = torch.sum(dist_all * mask_pos.float(), dim=1) / (torch.sum(dist_all * mask_neg.float(), dim=1) + 1e-8)

    # return the average of the negative log values
    return torch.mean(-torch.log(temp + 1e-8))
