import torch

from utils.metrics import pairwise_euclidean_distance


def kd_cluster_loss(embs_s, emb_size, lp_layer, labels, centroids, **kwargs):
    """
    Cluster-based Knowledge Distillation loss. It computes Davies-Bouldin index for each label in the batch.
    :param embs_s: embeddings from the student model for each item in the batch
    :param emb_size: size of the embeddings of the student model
    :param lp_layer: the linear projection layer
    :param labels: labels for each item in the batch
    :param centroids: centroids for all the labels from the training set. computed using the pre-trained MOVE model
    :param kwargs: any other arguments
    :return: average of the loss value for each label in the batch
    """
    # setting the proper device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # getting the unique labels of the batch
    unique_labels = torch.unique(labels)

    # computing teacher embeddings by passing the centroids of each label
    # to the linear projection layer
    embs_t = torch.cat([lp_layer(
        centroids[unique_labels[i].item()].to(device)) for i in range(unique_labels.size(0))], 0)

    # computing normalized Euclidean distance between student embeddings and each projected centroid
    dist_all = pairwise_euclidean_distance(embs_s, embs_t)
    dist_all /= emb_size

    # creating a mask for finding the related centroid for each item in the batch
    dist_mask = (pairwise_euclidean_distance(labels.view(-1, 1).double(),
                                             unique_labels.view(-1, 1).double()) < 0.5).float()

    # computing intra-cluster distances
    intra_dist = torch.mean(dist_all * dist_mask, dim=0)

    # computing inter-cluster distances with normalized Euclidean distance
    inter_dist = pairwise_euclidean_distance(embs_t)
    inter_dist /= emb_size

    # numerator of Davies-Bouldin index
    numerator = intra_dist.view(-1, 1) + intra_dist.view(1, -1)

    # computing Davies-Bouldin index
    tmp = (numerator / inter_dist) * (1 - torch.eye(numerator.size(0))).to(device)
    max_tmp = torch.max(tmp, dim=1).values

    return max_tmp.mean()
