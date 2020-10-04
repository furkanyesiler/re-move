import gc
import json
import os
import random
import time
from abc import ABC, abstractmethod

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.fixed_size_clique_dataset import FixedSizeCliqueDataset
from datasets.fixed_size_instance_dataset import FixedSizeInstanceDataset
from datasets.full_size_instance_dataset import FullSizeInstanceDataset
from utils.data_utils import handle_device
from utils.data_utils import import_dataset_from_pt
from utils.metrics import average_precision
from utils.metrics import pairwise_cosine_similarity
from utils.metrics import pairwise_euclidean_distance
from utils.metrics import pairwise_pearson_coef


DATASET_DICT = {0: FixedSizeCliqueDataset,
                1: FixedSizeInstanceDataset,
                2: FixedSizeInstanceDataset,
                3: FixedSizeCliqueDataset}


class BaseTrainer(ABC):
    """
    A Trainer object that serves as a template for the specific ones.
    """
    def __init__(self, cfg, experiment_name):
        """
        Initializing the object
        :param cfg: dictionary that holds the config hyper-parameters
        :param experiment_name: name of the experiment
        """
        # storing the config hyper-parameters and the name of the experiment
        self.cfg = cfg
        self.experiment_name = experiment_name

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True

        # setting the device for the tensor operations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # best mean average precision value. initialized at 0 for checkpointing.
        self.best_map = 0

        # initializing the required parameters of the Trainer object
        self.create_model()
        self.create_optimizer()
        self.create_loggers()
        self.create_lr_scheduler()

        # creating necessary directories for saving models and logs
        if not os.path.exists(os.path.join(self.cfg['main_path'], 'saved_models')):
            os.mkdir(os.path.join(self.cfg['main_path'], 'saved_models'))
        if not os.path.exists(os.path.join(self.cfg['main_path'], 'saved_models',
                                           '{}_models'.format(self.cfg['exp_type']))):
            os.mkdir(os.path.join(self.cfg['main_path'], 'saved_models',
                                  '{}_models'.format(self.cfg['exp_type'])))
        if not os.path.exists(os.path.join(self.cfg['main_path'], 'experiment_logs')):
            os.mkdir(os.path.join(self.cfg['main_path'], 'experiment_logs'))
        if not os.path.exists(os.path.join(self.cfg['main_path'], 'experiment_logs',
                                           '{}_logs'.format(self.cfg['exp_type']))):
            os.mkdir(os.path.join(self.cfg['main_path'], 'experiment_logs',
                                  '{}_logs'.format(self.cfg['exp_type'])))

    def train(self, save_logs=True):
        """
        Training function for one full training cycle. One full epoch is divided into
        4 mini-epochs for optimizing the speed and memory requirements of the training process.
        :param save_logs: whether to save training and validation loss logs
        """
        # iterating over the number of epochs
        for epoch in range(self.cfg['num_of_epochs']):
            # getting start time for logging
            start_time = time.monotonic()

            # updating the current epoch
            # it is useful for functions that require the current epoch information
            self.current_epoch = epoch

            train_loss_epoch = 0

            # dividing one full epoch to 4 mini-epochs
            for mini_epoch in range(4):
                self.mini_epoch = mini_epoch

                # creating a new data loader for the mini-epoch
                self.create_dataloader(is_train=True)

                # function that trains the model and returns the loss
                train_loss_epoch += self.handle_training_batches()

            # training loss values for 4 mini-epochs are averaged
            train_loss_epoch /= 4

            # logging the training loss
            self.train_loss_log.append(train_loss_epoch)

            # variable to track whether validation will be performed this epoch
            val_epoch = (epoch + 1) % 2 == 0 or epoch == self.cfg['num_of_epochs'] - 1
            # validation is performed every 2 epochs
            if val_epoch:
                # creating a new data loader for the validation set
                self.create_dataloader(is_train=False)

                # function for computing validation metrics
                self.handle_validation_batches()

                # checkpoint function to decide whether to save the current model
                self.checkpoint()

            # taking a step if a learning rate scheduler is used
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # computing the epoch time and printing the statistics
            epoch_time = (time.monotonic() - start_time)
            print_statement = 'Epoch {} --- Training loss: {:.5f}.'.format(self.current_epoch, self.train_loss_log[-1])
            print_statement += ' Validation MAP: {:.3f}.'.format(self.val_loss_log[-1]) if val_epoch else ''
            print_statement += ' Total time: {:.0f}m{:.0f}s.'.format(epoch_time // 60, epoch_time % 60)
            print(print_statement)

        if save_logs:
            with open('./experiment_logs/{}_logs/{}.json'.format(self.cfg['exp_type'], self.experiment_name), 'w') as f:
                json.dump({'train_loss_log': self.train_loss_log, 'val_loss_log': self.val_loss_log}, f)

    @abstractmethod
    def handle_training_batches(self):
        """
        Training loop for one mini-epoch. Each specific Trainer should implement this.
        """
        pass

    @abstractmethod
    def create_model(self):
        """
        Creating model. Every specific Trainer should implement this.
        """
        self.model = None

    @abstractmethod
    def create_optimizer(self):
        """
        Creating optimizer. Every specific Trainer should implement this
        due to change in parameters to give to the optimizer.
        """
        self.optimizer = None

    def create_lr_scheduler(self):
        """
        Creating a learning rate scheduler.
        """
        if self.cfg['lr_schedule'] != [0]:
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                               milestones=self.cfg['lr_schedule'],
                                                               gamma=self.cfg['lr_gamma'])
        else:
            self.lr_scheduler = None

    def create_loggers(self):
        """
        Creating logs for training and validation losses.
        """
        self.train_loss_log = []
        self.val_loss_log = []

    def handle_validation_batches(self):
        """
        Validation loop.
        """
        # setting the model to evaluation mode
        self.model.eval()

        # disabling gradient tracking
        with torch.no_grad():
            # initializing an empty tensor for storing the embeddings
            embed_all = torch.tensor([], device=self.device)

            # iterating through the data loader
            for batch_idx, items in enumerate(self.data_loader):
                # sending the items to the proper device
                items = handle_device(items, self.device)

                # forward pass of the model
                # obtaining the embeddings of each item in the batch
                embs = self.model(items)

                embed_all = torch.cat((embed_all, embs))

            # if Triplet or ProxyNCA loss is used, the distance function is Euclidean distance
            if self.cfg['loss'] in [0, 1]:
                dist_all = pairwise_euclidean_distance(embed_all)
                dist_all /= self.model.fin_emb_size
            # if NormalizedSoftmax loss is used, the distance function is cosine distance
            elif self.cfg['loss'] == 2:
                dist_all = -1 * pairwise_cosine_similarity(embed_all)
            # if Group loss is used, the distance function is Pearson correlation coefficient
            else:
                dist_all = -1 * pairwise_pearson_coef(embed_all)

        # computing evaluation metrics from the obtained distances
        val_map_score = average_precision(
            -1 * dist_all.cpu().float().clone() + torch.diag(torch.ones(len(self.data)) * float('-inf')),
            dataset=0, path=self.cfg['main_path'], print_metrics=False)

        # logging the mean average precision value of the current validation run
        self.val_loss_log.append(val_map_score.item())

    def checkpoint(self):
        """
        Function to determine whether to save the current model.
        If the current mean average precision is 0.001 higher than the best value, the model is saved.
        """
        if self.val_loss_log[-1] >= self.best_map + 1e-3:

            torch.save(self.model.state_dict(),
                       os.path.join(self.cfg['main_path'], 'saved_models', '{}_models'.format(self.cfg['exp_type']),
                                    'model_{}.pt'.format(self.experiment_name)))

            self.best_map = self.val_loss_log[-1]

    def create_dataloader(self, is_train=True):
        """
        Creating data loader for training and validation loops.
        :param is_train: whether the data loader is for a training loop
        """
        # flushing down the previous loaders and related variables
        self.data = None
        self.labels = None
        self.dataset = None
        self.data_loader = None
        gc.collect()

        if is_train:
            # creating chunks for mini-epochs
            self.create_chunks()

            # importing the selected chunks
            self.data, self.labels = import_dataset_from_pt(filename=self.cfg['train_file'],
                                                            chunks=self.chunks[self.mini_epoch], suffix=True)

            # creating the dataset object
            self.dataset = DATASET_DICT[self.cfg['loss']](self.data, self.labels,
                                                          self.cfg['patch_len'], self.cfg['data_aug'],
                                                          self.cfg['main_path'])

            # determining the batch size
            # it depends on the dataset/loss type
            batch_size = self.cfg['batch_size'] // 4 if self.cfg['loss'] in [0, 3] else self.cfg['batch_size']

            # creating the data loader
            self.data_loader = DataLoader(self.dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=self.cfg['num_workers'],
                                          collate_fn=self.dataset.collate, drop_last=True,
                                          pin_memory=False)

        else:
            # importing the validation data
            self.data, self.labels = import_dataset_from_pt(filename=self.cfg['val_file'], suffix=False)

            # creating the dataset object
            self.dataset = FullSizeInstanceDataset(self.data)

            # creating the data loader
            self.data_loader = DataLoader(self.dataset, batch_size=1,
                                          shuffle=False, num_workers=self.cfg['num_workers'])

    def create_chunks(self):
        """
        Creating chunks of training data for each epoch.
        The training data is stored in 17 chunks.
        """
        if self.mini_epoch == 0:
            temp = list(range(1, 17))
            random.shuffle(temp)
            self.chunks = [list(temp[i*4:(i+1)*4]) for i in range(4)]
            self.chunks[3].append(17)
