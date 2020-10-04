import torch.nn as nn

from utils.metrics import pairwise_cosine_similarity


def normalizedsoftmax_loss(data, labels, proxies, **kwargs):
    """
    NormalizedSoftmax loss function. It uses cosine similarity and has a temperature parameter (0.05).
    :param data: embeddings of each item in the batch
    :param labels: labels of each item in the batch
    :param proxies: proxy vectors for each class
    :param kwargs: any other arguments
    :return: average loss value for the batch
    """
    # compute pairwise cosine similarity
    dist_all = pairwise_cosine_similarity(data, proxies)

    # return the loss value for the batch
    return nn.CrossEntropyLoss()(dist_all / 0.05, labels.long().squeeze())
