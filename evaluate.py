import os
import time

import torch
from torch.utils.data import DataLoader

from datasets.full_size_instance_dataset import FullSizeInstanceDataset
from models.move_model import MOVEModel
from utils.metrics import average_precision
from utils.metrics import pairwise_cosine_similarity
from utils.metrics import pairwise_euclidean_distance
from utils.metrics import pairwise_pearson_coef
from utils.data_utils import import_dataset_from_pt
from utils.data_utils import handle_device


def evaluate(exp_name,
             exp_type,
             main_path,
             emb_size,
             loss
             ):
    """
    Main evaluation function of MOVE. For a detailed explanation of parameters,
    please check 'python move_main.py -- help'
    :param main_path: main working directory
    :param exp_name: name to save model and experiment summary
    :param exp_type: type of experiment
    :param emb_size: the size of the final embeddings produced by the model
    :param loss: the loss used for training the model
    """

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    eval_dataset = os.path.join(main_path, 'data/benchmark_crema.pt')

    print('Evaluating model {} on dataset {}.'.format(exp_name, eval_dataset))

    # initializing the model
    model = MOVEModel(emb_size=emb_size)

    # loading a pre-trained model
    model_name = os.path.join(main_path, 'saved_models', '{}_models'.format(exp_type), 'model_{}.pt'.format(exp_name))
    model.load_state_dict(torch.load(model_name, map_location='cpu'))

    # sending the model to gpu, if available
    model.to(device)

    # loading test data, initializing the dataset object and the data loader
    test_data, test_labels = import_dataset_from_pt(filename=eval_dataset, suffix=False)
    test_set = FullSizeInstanceDataset(data=test_data)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    start_time = time.monotonic()

    with torch.no_grad():  # disabling gradient tracking
        model.eval()  # setting the model to evaluation mode

        # initializing an empty tensor for storing the embeddings
        embed_all = torch.tensor([], device=device)

        # iterating through the data loader
        for batch_idx, item in enumerate(test_loader):
            # sending the items to the proper device
            item = handle_device(item, device)

            # forward pass of the model
            # obtaining the embeddings of each item in the batch
            emb = model(item)

            # appending the current embedding to the collection of embeddings
            embed_all = torch.cat((embed_all, emb))

        # if Triplet or ProxyNCA loss is used, the distance function is Euclidean distance
        if loss in [0, 1]:
            dist_all = pairwise_euclidean_distance(embed_all)
            dist_all /= model.fin_emb_size
        # if NormalizedSoftmax loss is used, the distance function is cosine distance
        elif loss == 2:
            dist_all = -1 * pairwise_cosine_similarity(embed_all)
        # if Group loss is used, the distance function is Pearson correlation coefficient
        else:
            dist_all = -1 * pairwise_pearson_coef(embed_all)

    # computing evaluation metrics from the obtained distances
    average_precision(
        -1 * dist_all.cpu().float().clone() + torch.diag(torch.ones(len(test_data)) * float('-inf')), dataset=1)

    test_time = time.monotonic() - start_time

    print('Total time: {:.0f}m{:.0f}s.'.format(test_time // 60, test_time % 60))
