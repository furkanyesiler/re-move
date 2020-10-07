import torch

from utils.metrics import pairwise_euclidean_distance


def create_label_masks(labels):
    """
    Creates positive and negatives masks for given list of labels
    :param labels: labels to use for creating the masks
    :return: positive and negative masks
    """
    mask_diag = (1 - torch.eye(labels.size(0))).long()
    if torch.cuda.is_available():
        labels = labels.cuda()
        mask_diag = mask_diag.cuda()
    temp_mask = (pairwise_euclidean_distance(labels.double()) < 0.5).long()
    mask_pos = mask_diag * temp_mask
    mask_neg = mask_diag * (1 - mask_pos)

    return mask_pos, mask_neg


def triplet_mining_hard(dist_all, mask_pos, mask_neg):
    """
    Performs online hard mining
    :param dist_all: pairwise distance matrix
    :param mask_pos: mask for positive elements of triplets
    :param mask_neg: mask for negative elements of triplets
    :return: selected positive and negative distances
    """
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # selecting the positive elements of triplets
    _, sel_pos = torch.max(dist_all * mask_pos.float(), 1)
    dists_pos = torch.gather(dist_all, 1, sel_pos.view(-1, 1))

    # modifying the negative mask for hard mining
    mask_neg = torch.where(mask_neg == 0, torch.tensor(float('inf'), device=device),
                           torch.tensor(1., device=device))

    # selecting the negative elements of triplets
    _, sel_neg = torch.min(dist_all + mask_neg.float(), 1)
    dists_neg = torch.gather(dist_all, 1, sel_neg.view(-1, 1))

    return dists_pos, dists_neg


def triplet_mining_random(dist_all, mask_pos, mask_neg):
    """
    Performs online random mining
    :param dist_all: pairwise distance matrix
    :param mask_pos: mask for positive elements of triplets
    :param mask_neg: mask for negative elements of triplets
    :return: selected positive and negative distances
    """
    # selecting the positive elements of triplets
    _, sel_pos = torch.max(mask_pos.float() + torch.rand_like(dist_all), 1)
    dists_pos = torch.gather(dist_all, 1, sel_pos.view(-1, 1))

    # selecting the negative elements of triplets
    _, sel_neg = torch.max(mask_neg.float() + torch.rand_like(dist_all), 1)
    dists_neg = torch.gather(dist_all, 1, sel_neg.view(-1, 1))

    return dists_pos, dists_neg


def triplet_mining_semihard(dist_all, mask_pos, mask_neg):
    """
    Performs online semi-hard mining
    :param dist_all: pairwise distance matrix
    :param mask_pos: mask for positive elements of triplets
    :param mask_neg: mask for negative elements of triplets
    :return: selected positive and negative distances
    """
    # selecting the positive elements of triplets
    _, sel_pos = torch.max(mask_pos.float() + torch.rand_like(dist_all), 1)
    dists_pos = torch.gather(dist_all, 1, sel_pos.view(-1, 1))

    # selecting the negative elements of triplets
    _, sel_neg = torch.max((mask_neg + mask_neg * (dist_all < dists_pos.expand_as(dist_all)).long()).float() +
                           torch.rand_like(dist_all), 1)
    dists_neg = torch.gather(dist_all, 1, sel_neg.view(-1, 1))

    return dists_pos, dists_neg


def triplet_mining_allpos_semihard(dist_all, mask_pos, mask_neg):
    """
    Performs online semi-hard mining
    :param dist_all: pairwise distance matrix
    :param mask_pos: mask for positive elements of triplets
    :param mask_neg: mask for negative elements of triplets
    :return: selected positive and negative distances
    """
    mask_pos = mask_pos.clone()
    dist_pos_all = torch.Tensor([])
    dist_neg_all = torch.Tensor([])

    if torch.cuda.is_available():
        dist_pos_all = dist_pos_all.cuda()
        dist_neg_all = dist_neg_all.cuda()

    for i in range(3):
        _, sel_pos = torch.max(mask_pos.float(), 1)
        dists_pos = torch.gather(dist_all, 1, sel_pos.view(-1, 1))
        mask_pos.scatter_(1, sel_pos.view(-1, 1), 0)

        # selecting the negative elements of triplets
        _, sel_neg = torch.max((mask_neg + mask_neg * (dist_all < dists_pos.expand_as(dist_all)).long()).float() +
                               torch.rand_like(dist_all), 1)
        dists_neg = torch.gather(dist_all, 1, sel_neg.view(-1, 1))

        dist_pos_all = torch.cat((dist_pos_all, dists_pos))
        dist_neg_all = torch.cat((dist_neg_all, dists_neg))

    return dist_pos_all, dist_neg_all
