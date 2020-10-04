import torch

from utils.metrics import pairwise_euclidean_distance


def kd_distance_loss(embs_s, embs_t, emb_size, **kwargs):
    """
    Distance-based Knowledge Distillation loss. It computes the L1 loss between the distances obtained
    from the student model and the ones from the teacher model
    :param embs_s: embeddings from the student model for each item in the batch
    :param embs_t: embeddings from the student model for each item in the batch
    :param emb_size: size of the embeddings of the student model
    :param kwargs: any other arguments
    :return:
    """
    # getting non-diagonal indices
    idxs = (1 - torch.eye(embs_s.size(0))).nonzero()

    # sending the indices to cuda if needed
    if torch.cuda.is_available():
        idxs = idxs.cuda()

    # computing pairwise Euclidean distances for both the student and the teacher embeddings
    dist_all_s = pairwise_euclidean_distance(embs_s)
    dist_all_t = pairwise_euclidean_distance(embs_t)

    dist_all_s /= emb_size
    dist_all_t /= 16000

    # leaving the self-distances out
    dist_all_s = dist_all_s[idxs[:, 0], idxs[:, 1]]
    dist_all_t = dist_all_t[idxs[:, 0], idxs[:, 1]]

    # compute and return L1Loss
    return torch.nn.L1Loss()(dist_all_s, dist_all_t)
