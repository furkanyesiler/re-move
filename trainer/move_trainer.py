import numpy as np
import torch

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


class MOVETrainer(BaseTrainer):
    """
    Trainer object for baseline experiments with MOVE.
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
        # creating the model and sending it to the proper device
        self.model = MOVEModel(emb_size=self.cfg['emb_size'])
        self.model.to(self.device)

        # computing and printing the total number of parameters of the model
        self.num_params = 0
        for param in self.model.parameters():
            self.num_params += np.prod(param.size())
        print('Total number of parameters for the model: {:.0f}'.format(self.num_params))

    def create_optimizer(self):
        """
        Initializing the optimizer.
        """
        # parameters to train
        opt_params = list(self.model.parameters())

        # initializing proxies if a proxy-based loss is used
        self.proxies = None
        if self.cfg['loss'] in [1, 2, 3]:
            self.proxies = torch.nn.Parameter(torch.randn(14499, self.cfg['emb_size'],
                                                          requires_grad=True, device=self.device))
            opt_params.append(self.proxies)

        if self.cfg['optimizer'] == 0:
            self.optimizer = torch.optim.SGD(opt_params,
                                             lr=self.cfg['learning_rate'],
                                             momentum=self.cfg['momentum'])
        elif self.cfg['optimizer'] == 1:
            self.optimizer = Ranger(opt_params,
                                    lr=self.cfg['learning_rate'])
        else:
            self.optimizer = None
