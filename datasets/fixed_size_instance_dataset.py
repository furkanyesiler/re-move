import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_utils import cs_augment


class FixedSizeInstanceDataset(Dataset):
    """
    This dataset object returns 1 song.
    Given features are pre-processed to have a same particular shape.
    """

    def __init__(self, data, labels, w=1800, data_aug=1, path=''):
        """
        Initializing the dataset object
        :param data: pcp features
        :param labels: labels of features (should be in the same order as features)
        :param w: width of pcp features (number of frames in the temporal dimension)
        :param data_aug: whether to apply data augmentation to each song (1 or 0)
        :param path: the path of the working directory
        """
        self.data = data  # pcp features
        self.labels = labels  # labels of the pcp features

        self.h = 23  # height of pcp features
        self.w = w  # width of pcp features
        self.data_aug = data_aug  # whether to apply data augmentation to each song

        # dictionary to store one unique integer for each label
        self.label_to_unique_int = torch.load(os.path.join(path, 'data/red_label_idx.pt'))

    def __getitem__(self, index):
        """
        Getting the pcp feature and label for the selected song
        :param index: index of the song picked by the data loader
        :return: pcp feature and label from the picked song
        """

        label = self.labels[index]  # getting the clique chosen by the data loader
        item = self.data[index]

        if self.data_aug == 1:  # applying data augmentation to the song
            item = cs_augment(item)
        # if the song is longer than the required width, choose a random start point to crop
        if item.shape[2] >= self.w:
            p_index = [i for i in range(0, item.shape[2] - self.w + 1)]
            if len(p_index) != 0:
                start = np.random.choice(p_index)
                item = item[:, :, start:start + self.w]
        else:  # if the song is shorter than the required width, zero-pad the end
            item = torch.cat((item, torch.zeros([1, self.h, self.w - item.shape[2]])), 2)

        return item, torch.tensor([self.label_to_unique_int[label]])

    def __len__(self):
        """
        Length of the dataset object
        :return: length of the data list containing all the songs
        """
        return len(self.data)

    @staticmethod
    def collate(batch):
        """
        Custom collate function for this dataset object
        :param batch: elements of the mini-batch (pcp features and labels)
        :return: collated elements
        """
        items = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        return torch.stack(items, 0), torch.stack(labels, 0)
