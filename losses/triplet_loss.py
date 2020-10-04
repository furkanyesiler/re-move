import torch.nn as nn

from utils.loss_utils import create_label_masks
from utils.loss_utils import triplet_mining_random
from utils.loss_utils import triplet_mining_semihard
from utils.loss_utils import triplet_mining_hard
from utils.loss_utils import triplet_mining_allpos_semihard
from utils.metrics import pairwise_euclidean_distance


MINING_STRATEGIES = [triplet_mining_random,
                     triplet_mining_semihard,
                     triplet_mining_hard,
                     triplet_mining_allpos_semihard]


def triplet_loss(data, labels, emb_size, margin, mining_strategy, **kwargs):
    """
    Triplet loss function. It includes triplet mining and returns the loss as the
    average of all the triplets.
    :param data: embeddings of each item in the batch
    :param labels: labels of each item in the batch
    :param emb_size: size of the embeddings for normalization
    :param margin: margin used for triplet loss
    :param mining_strategy: mining strategy to use
    :param kwargs: any other arguments
    :return: average of the loss values for all the selected triplets
    """
    mask_pos, mask_neg = create_label_masks(labels)

    dist_all = pairwise_euclidean_distance(data)
    dist_all /= emb_size

    dist_g, dist_i = MINING_STRATEGIES[mining_strategy](dist_all, mask_pos, mask_neg)

    loss = nn.functional.relu(dist_g + (margin - dist_i))

    return loss.mean()
