import json
import os

import numpy as np
import torch

from losses.triplet_loss import triplet_loss
from models.move_model import MOVEModel
from trainer.base_trainer import BaseTrainer
from utils.data_utils import handle_device
from utils.ranger import Ranger


NUM_OF_ROWS_TO_PRUNE = [8000, 12000, 13952, 14976, 15488, 15744, 15872]


class PruningTrainer(BaseTrainer):
    """
    Trainer object for Pruning experiments.
    """
    def __init__(self, cfg, experiment_name):
        """
        Initializing the trainer
        :param cfg: dictionary that holds the config hyper-parameters
        :param experiment_name: name of the experiment
        """
        # initializing the parent Trainer object
        super().__init__(cfg, experiment_name)

    def train(self, save_logs=True):
        """
        Main training function for Pruning experiments.
        It overrides the training function of the BaseTrainer for adding
        pruning-related functionality.
        :param save_logs: whether to save training and validation loss logs
        """
        # save the initial parameters of the model for other pruning iterations
        torch.save(self.model.state_dict(),
                   os.path.join(self.cfg['main_path'], 'saved_models', 'pruning_models',
                                'model_{}_initial.pt'.format(self.experiment_name)))

        # iterating full-training cycles for pruning
        for prune_iteration in range(self.cfg['pruning_iterations'] + 1):
            self.prune_iteration = prune_iteration

            # loading the initial parameters of the model
            if prune_iteration > 0:
                self.model.load_state_dict(torch.load(os.path.join(self.cfg['main_path'],
                                                                   'saved_models',
                                                                   'pruning_models',
                                                                   'model_{}_initial.pt'.format(self.experiment_name))))

                # resetting the learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.cfg['learning_rate']

                # re-creating the learning rate schedule for the new training cycle
                self.create_lr_scheduler()

            # execute a full training cycle
            super().train(save_logs=False)

            # selecting which indices of the linear layer to prune
            # based on the trained model
            self.select_indices_to_prune()

        if save_logs:
            with open('./experiment_logs/{}_logs/{}.json'.format(self.cfg['exp_type'], self.experiment_name), 'w') as f:
                json.dump({'train_loss_log': self.train_loss_log, 'val_loss_log': self.val_loss_log}, f)

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
            loss = triplet_loss(data=embs, labels=labels, emb_size=self.cfg['emb_size'],
                                margin=self.cfg['margin'], mining_strategy=self.cfg['mining_strategy'])

            # setting gradients of the optimizer to zero
            self.optimizer.zero_grad()

            # calculating gradients with backpropagation
            loss.backward()

            # updating the weights
            self.optimizer.step()

            # applying the zero-mask to the selected indices
            if self.prune_iteration > 0:
                self.apply_mask()

            # logging the loss value of the current batch
            epoch_loss.append(loss.detach().item())

        # logging the loss value of the current mini-epoch
        return np.mean(epoch_loss)

    def apply_mask(self):
        """
        Applying the mask tensor to the linear layer to 'prune' weights.
        """
        self.model.lin1.weight.data = self.model.lin1.weight.data * self.mask
        self.model.fin_emb_size = self.model.lin1.weight.shape[0] - NUM_OF_ROWS_TO_PRUNE[self.prune_iteration]

    def select_indices_to_prune(self):
        """
        Selecting which indices to prune based on the trained model.
        :return:
        """
        self.indices_to_prune = torch.topk(torch.abs(self.model.lin1.weight).mean(dim=1),
                                           k=NUM_OF_ROWS_TO_PRUNE[self.prune_iteration], largest=False).indices

        # creating a mask of ones and zeros
        mask = torch.ones(self.model.lin1.weight.shape)
        zero_row = torch.zeros(1, self.model.lin1.weight.shape[1])

        # sending the tensors to the proper device
        mask = handle_device(mask, self.device)
        zero_row = handle_device(zero_row, self.device)

        # finalizing the mask based on the selected indices
        mask[self.indices_to_prune] = zero_row
        self.mask = mask

    def create_model(self):
        """
        Initializing the model to optimize.
        """
        # creating the model and sending it to the proper device
        self.model = MOVEModel(emb_size=16000)
        self.model.to(self.device)

        # computing and printing the total number of parameters of the new model
        self.num_params = 0
        for param in self.model.parameters():
            self.num_params += np.prod(param.size())
        print('Total number of parameters for the model: {:.0f}'.format(self.num_params))

    def create_optimizer(self):
        """
        Initializing the optimizer.
        """
        if self.cfg['optimizer'] == 0:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.cfg['learning_rate'],
                                             momentum=self.cfg['momentum'])
        elif self.cfg['optimizer'] == 1:
            self.optimizer = Ranger(self.model.parameters(),
                                    lr=self.cfg['learning_rate'])
        else:
            self.optimizer = None
