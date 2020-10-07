import os

import numpy as np
import torch
import torch.nn as nn

from losses.kd_distance_loss import kd_distance_loss
from losses.kd_cluster_loss import kd_cluster_loss
from losses.triplet_loss import triplet_loss
from models.move_model import MOVEModel
from trainer.base_trainer import BaseTrainer
from utils.data_utils import handle_device
from utils.ranger import Ranger


KD_LOSS_DICT = {'distance': kd_distance_loss,
                'cluster': kd_cluster_loss}


class KDTrainer(BaseTrainer):
    """
    Trainer object for Knowledge Distillation experiments.
    """
    def __init__(self, cfg, experiment_name):
        """
        Initializing the trainer
        :param cfg: dictionary that holds the config hyper-parameters
        :param experiment_name: name of the experiment
        """
        # initializing the parent Trainer object
        super().__init__(cfg, experiment_name)

    def handle_training_batches(self):
        """
        Training loop for one mini-epoch.
        :return: training loss for the current mini-epoch
        """
        # setting the model to training mode
        self.model.train()

        # initializing a list object to hold losses from each iteration
        epoch_loss = []

        # training loop
        for batch_idx, batch in enumerate(self.data_loader):
            # if overfit_batch == 1, only the same batch is trained.
            # this helps to see whether there are any issues with optimization.
            # a fast over-fitting behaviour is expected.
            if self.cfg['overfit_batch'] == 1:
                if batch_idx == 0:
                    overfit_batch = batch
                else:
                    batch = overfit_batch

            # making sure the data and labels are in the correct device and in float32 type
            items, labels = batch
            items = handle_device(items, self.device)
            labels = handle_device(labels, self.device)

            # forward pass of the student model
            # obtaining the embeddings of each item in the batch
            embs_s = self.model(items)

            # if the distance-based KD loss is chosen,
            # we obtain the embeddings of each item from the teacher model
            with torch.no_grad():
                embs_t = self.teacher(items) if self.cfg['kd_loss'] == 'distance' else None

            # calculating the KD loss for the iteration
            kd_loss = KD_LOSS_DICT[self.cfg['kd_loss']](embs_s=embs_s, embs_t=embs_t, emb_size=self.cfg['emb_size'],
                                                        lp_layer=self.lp_layer, labels=labels, centroids=self.centroids)

            # calculating the triplet loss for the iteration
            main_loss = triplet_loss(data=embs_s, labels=labels, emb_size=self.cfg['emb_size'],
                                     margin=self.cfg['margin'], mining_strategy=self.cfg['mining_strategy'])

            # summing KD and triplet loss values
            loss = kd_loss + main_loss

            # setting gradients of the optimizer to zero
            self.optimizer.zero_grad()

            # calculating gradients with backpropagation
            loss.backward()

            # updating the weights
            self.optimizer.step()

            # logging the loss value of the current batch
            epoch_loss.append(loss.detach().item())

        # logging the loss value of the current mini-epoch
        return np.mean(epoch_loss)

    def create_model(self):
        """
        Initializing the model to optimize.
        """
        # creating the student model and sending it to the proper device
        self.model = MOVEModel(emb_size=self.cfg['emb_size'], sum_method=4, final_activation=3)
        self.model.to(self.device)

        # initializing necessary models/data for KD training
        self.teacher = None
        self.lp_layer = None
        self.centroids = None

        # creating the teacher model and sending it to the proper device
        # this step is for the distance-based KD training
        if self.cfg['kd_loss'] == 'distance':
            self.teacher = MOVEModel(emb_size=16000, sum_method=4, final_activation=3)
            self.teacher.load_state_dict(torch.load(os.path.join(self.cfg['main_path'],
                                                                 'saved_models/model_move.pt'),
                                                    map_location='cpu'))
            self.teacher.to(self.device)
            self.teacher.eval()

        # creating the linear projection layer and loading the class centroids
        # this step is for the cluster-based KD training
        elif self.cfg['kd_loss'] == 'cluster':
            self.lp_layer = nn.Linear(in_features=16000, out_features=self.cfg['emb_size'], bias=False)
            self.lp_layer.to(self.device)
            self.centroids = torch.load(os.path.join(self.cfg['main_path'], 'data/centroids.pt'))

        # computing and printing the total number of parameters of the new model
        self.num_params = 0
        for param in self.model.parameters():
            self.num_params += np.prod(param.size())
        print('Total number of parameters for the model: {:.0f}'.format(self.num_params))

    def create_optimizer(self):
        """
        Initializing the optimizer.
        In the case of distance-based KD training, no additional parameters are given to the optimizer.
        In the case of cluster-based KD training, the parameters of the linear projection layer are updated,
        as well as the parameters of the student model.
        """
        # getting the parameters of the student model
        opt_params = list(self.model.parameters())

        # for the cluster-based KD training, append the parameters of
        # the linear projection layer for the optimizer
        if self.cfg['kd_loss'] == 'cluster':
            opt_params += list(self.lp_layer.parameters())

        if self.cfg['optimizer'] == 0:
            self.optimizer = torch.optim.SGD(opt_params,
                                             lr=self.cfg['learning_rate'],
                                             momentum=self.cfg['momentum'])
        elif self.cfg['optimizer'] == 1:
            self.optimizer = Ranger(opt_params,
                                    lr=self.cfg['learning_rate'])
        else:
            self.optimizer = None
