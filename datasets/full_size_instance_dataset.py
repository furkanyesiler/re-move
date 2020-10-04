from torch.utils.data import Dataset


class FullSizeInstanceDataset(Dataset):
    """
    MOVEDataset object returns one song from the test data.
    Given features are in their full length.
    """

    def __init__(self, data):
        """
        Initialization of the dataset object
        :param data: pcp features
        """
        self.data = data  # pcp features

    def __getitem__(self, index):
        """
        Getting the pcp feature of a song
        :param index: index of the song picked by the data loader
        :return: pcp feature of the selected song
        """

        item = self.data[index]

        return item.float()

    def __len__(self):
        """
        Length of the entire dataset
        :return: length of the entire dataset
        """
        return len(self.data)
