import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_utils import cs_augment


class FixedSizeCliqueDataset(Dataset):
    """
    This dataset object returns 4 songs for a given label.
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
        self.labels = np.array(labels)  # labels of the pcp features

        self.h = 23  # height of pcp features
        self.w = w  # width of pcp features
        self.data_aug = data_aug  # whether to apply data augmentation to each song

        self.labels_set = set(self.labels)  # the set of labels

        # dictionary to store which indexes belong to which label
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

        # dictionary to store one unique integer for each label
        self.label_to_unique_int = torch.load(os.path.join(path, 'data/red_label_idx.pt'))

        self.clique_list = []  # list to store all cliques

        # adding some cliques multiple times depending on their size
        for label in self.label_to_indices.keys():
            if self.label_to_indices[label].size < 2:
                continue
            if self.label_to_indices[label].size < 6:
                self.clique_list.extend([label] * 1)
            elif self.label_to_indices[label].size < 10:
                self.clique_list.extend([label] * 2)
            elif self.label_to_indices[label].size < 14:
                self.clique_list.extend([label] * 3)
            else:
                self.clique_list.extend([label] * 4)

    def __getitem__(self, index):
        """
        Getting the pcp features and labels for 4 songs of a label
        :param index: index of the clique picked by the data loader
        :return: 4 songs and their labels from the picked label
        """

        label = self.clique_list[index % len(self.clique_list)]  # getting the label chosen by the data loader

        if self.label_to_indices[label].size == 2:  # if the clique size is 2, repeat the already selected songs
            idx1, idx2 = np.random.choice(self.label_to_indices[label], 2, replace=False)
            item1, item2 = self.data[idx1], self.data[idx2]
            item3, item4 = self.data[idx1], self.data[idx2]
        elif self.label_to_indices[label].size == 3:  # if the clique size is 3, choose one of the songs twice
            idx1, idx2, idx3 = np.random.choice(self.label_to_indices[label], 3, replace=False)
            idx4 = np.random.choice(self.label_to_indices[label], 1, replace=False)[0]
            item1, item2, item3, item4 = self.data[idx1], self.data[idx2], self.data[idx3], self.data[idx4]
        else:  # if the clique size is larger than or equal to 4, choose 4 songs randomly
            idx1, idx2, idx3, idx4 = np.random.choice(self.label_to_indices[label], 4, replace=False)
            item1, item2, item3, item4 = self.data[idx1], self.data[idx2], self.data[idx3], self.data[idx4]
        items_i = [item1, item2, item3, item4]  # list for storing selected songs

        items = []

        # pre-processing each song separately
        for item in items_i:
            if self.data_aug == 1:  # applying data augmentation to the song
                item = cs_augment(item)
            # if the song is longer than the required width, choose a random start point to crop
            if item.shape[2] >= self.w:
                p_index = [i for i in range(0, item.shape[2] - self.w + 1)]
                if len(p_index) != 0:
                    start = np.random.choice(p_index)
                    temp_item = item[:, :, start:start + self.w]
                    items.append(temp_item)
            else:  # if the song is shorter than the required width, zero-pad the end
                items.append(torch.cat((item, torch.zeros([1, self.h, self.w - item.shape[2]])), 2))

        return torch.stack(items, 0), torch.stack([torch.tensor([self.label_to_unique_int[label]])] * 4, 0)

    def __len__(self):
        """
        Length of the dataset object
        :return: length of the clique list containing all the labels (multiple cliques included for larger ones)
        """
        return len(self.clique_list)

    @staticmethod
    def collate(batch):
        """
        Custom collate function for this dataset object
        :param batch: elements of the mini-batch (pcp features and labels)
        :return: collated elements
        """
        items = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        return torch.cat(items, 0), torch.cat(labels, 0)
