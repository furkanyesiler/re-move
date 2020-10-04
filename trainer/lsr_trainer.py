import os

import numpy as np
import torch
import torch.nn as nn

from losses.group_loss import group_loss
from losses.normalizedsoftmax_loss import normalizedsoftmax_loss
from losses.proxynca_loss import proxynca_loss
from losses.triplet_loss import triplet_loss
from models.move_model import MOVEModel
from trainer.base_trainer import BaseTrainer
from utils.data_utils import handle_device
from utils.ranger import Ranger


LOSS_DICT = {0: triplet_loss,
             1: proxynca_loss,
             2: normalizedsoftmax_loss,
             3: group_loss}


class LSRTrainer(BaseTrainer):
    """
    Trainer object for Latent Space Reconfiguration experiments.
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
        :return training loss for the current mini-epoch
        """
        # setting the model to training mode
        self.model.train()

        # initializing a list object to hold losses from each iteration
        epoch_loss = []

        # for the first epoch, only the linear layer is trained.
        # starting from the second epoch, all the parameters of the model are trained.
        if self.current_epoch == 1:
            for param in self.model.parameters():
                param.requires_grad = True

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

            # forward pass of the model
            # obtaining the embeddings of each item in the batch
            embs = self.model(items)

            # calculating the loss value for the iteration
            loss = LOSS_DICT[self.cfg['loss']](data=embs, labels=labels, emb_size=self.model.fin_emb_size,
                                               proxies=self.proxies, margin=self.cfg['margin'],
                                               mining_strategy=self.cfg['mining_strategy'])

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
        # creating and loading the learned parameters of the MOVE model
        # this model stands as our base model
        self.model = MOVEModel(emb_size=16000, sum_method=4, final_activation=3)
        self.model.load_state_dict(torch.load(os.path.join(self.cfg['main_path'],
                                                           'saved_models/model_move.pt'),
                                              map_location='cpu'))

        # freezing the parameters of all the parameters of the base model
        for param in self.model.parameters():
            param.requires_grad = False

        # creating a new linear layer and a new batch normalization layer
        self.model.lin1 = torch.nn.Linear(in_features=256, out_features=self.cfg['emb_size'], bias=False)
        self.model.lin_bn = torch.nn.BatchNorm1d(self.cfg['emb_size'], affine=False)

        # setting the embedding size of the model
        self.model.fin_emb_size = self.cfg['emb_size']

        # sending the model to the proper device
        self.model.to(self.device)

        # computing and printing the total number of parameters of the new model
        self.num_params = 0
        for param in self.model.parameters():
            self.num_params += np.prod(param.size())
        print('Total number of parameters for the model: {:.0f}'.format(self.num_params))

    def create_optimizer(self):
        """
        Initializing the optimizer. For LSR training, we have two types of parameters.
        'new_param' are the ones from the new linear layer,
        and 'finetune_param' are the ones from the 'feature extractor' part of MOVE model.
        By distinguishing them, we can set different learning rates for each parameter group.
        """
        # getting parameter groups as explained above
        param_list = ['lin1.weight', 'lin1.bias']
        new_param = [par[1] for par in self.model.named_parameters() if par[0] in param_list]
        finetune_param = [par[1] for par in self.model.named_parameters() if par[0] not in param_list]

        # initializing proxies if a proxy-based loss is used
        self.proxies = None
        if self.cfg['loss'] in [1, 2, 3]:
            self.proxies = torch.nn.Parameter(torch.randn(14499, self.cfg['emb_size'],
                                                          requires_grad=True, device=self.device))
            new_param.append(self.proxies)

        # setting the proper learning rates and initializing the optimizer
        opt_params = [{'params': finetune_param, 'lr': self.cfg['finetune_learning_rate']},
                      {'params': new_param}]

        if self.cfg['optimizer'] == 0:
            self.optimizer = torch.optim.SGD(opt_params,
                                             lr=self.cfg['learning_rate'],
                                             momentum=self.cfg['momentum'])
        elif self.cfg['optimizer'] == 1:
            self.optimizer = Ranger(opt_params,
                                    lr=self.cfg['learning_rate'])
        else:
            self.optimizer = None
