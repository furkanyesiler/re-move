import torch
import numpy as np
from scipy import interpolate


def import_dataset_from_pt(filename, chunks=17, suffix=True):
    """
    loading a dataset stored in .pt format
    :param filename: name of the .pt file to load
    :param chunks: number of chunks to load
    :param suffix: whether to load files in the following convention: 'filename_N.pt'
    :return: lists that contain data and labels (elements are in the same order)
    """
    if suffix:
        for i in chunks:
            dataset_dict = torch.load('{}_{}.pt'.format(filename, i))
            if i == chunks[0]:
                data = dataset_dict['data']
                labels = dataset_dict['labels']
            else:
                data.extend(dataset_dict['data'])
                labels.extend(dataset_dict['labels'])
    else:
        dataset_dict = torch.load('{}'.format(filename))
        data = dataset_dict['data']
        labels = dataset_dict['labels']

    return data, labels


def cs_augment(pcp, p_pitch=1, p_stretch=0.3, p_warp=0.3):
    """
    applying data augmentation to a given pcp patch
    :param pcp: pcp patch to augment (dimensions should be 1 x H x W)
    :param p_pitch: probability of applying pitch transposition
    :param p_stretch: probability of applying time stretch (with linear interpolation)
    :param p_warp: probability of applying time warping (silence, duplication, removal)
    :return: augmented pcp patch
    """
    pcp = pcp.cpu().detach().numpy()  # converting the pcp patch to a numpy matrix

    # pitch transposition
    if torch.rand(1) <= p_pitch:
        shift_amount = 0
        while shift_amount == 0:  # choosing the number of bins to roll
            shift_amount = torch.randint(low=0, high=12, size=(1,))
        pcp_aug = np.roll(pcp, shift_amount, axis=1)  # applying pitch transposition
    else:
        pcp_aug = pcp

    _, h, w = pcp_aug.shape
    times = np.arange(0, w)  # the original time stamps

    # interpolation function for time stretching and warping
    func = interpolate.interp1d(times, pcp_aug, kind='nearest', fill_value='extrapolate')

    # time stretch
    if torch.rand(1) < p_stretch:
        p = torch.rand(1)  # random number to determine the factor of time stretching
        if p <= 0.5:
            times_aug = np.linspace(0, w - 1, int(w * torch.clamp((1 - p), min=0.7, max=1).item()))
        else:
            times_aug = np.linspace(0, w - 1, int(w * torch.clamp(2 * p, min=1, max=1.5).item()))
        pcp_aug = func(times_aug)  # applying time stretching
    else:
        times_aug = times
        pcp_aug = func(times_aug)

    # time warping
    if torch.rand(1) < p_warp:
        p = torch.rand(1)  # random number to determine which operation to apply for time warping

        if p < 0.3:  # silence
            # each frame has a probability of 0.1 to be silenced
            silence_idxs = np.random.choice([False, True], size=times_aug.size, p=[.9, .1])
            pcp_aug[:, :, silence_idxs] = np.zeros((h, 1))

        elif p < 0.7:  # duplicate
            # each frame has a probability of 0.15 to be duplicated
            duplicate_idxs = np.random.choice([False, True], size=times_aug.size, p=[.85, .15])
            times_aug = np.sort(np.concatenate((times_aug, times_aug[duplicate_idxs])))
            pcp_aug = func(times_aug)

        else:  # remove
            # each frame has a probability of 0.1 to be removed
            remaining_idxs = np.random.choice([False, True], size=times_aug.size, p=[.1, .9])
            times_aug = times_aug[remaining_idxs]
            pcp_aug = func(times_aug)

    return torch.from_numpy(pcp_aug)  # casting the augmented pcp patch as a torch tensor


def handle_device(tensor, device):
    """
    Helper function to make sure the tensors are in the correct device
    for training and in correct precision.
    :param tensor: tensor to handle
    :param device: 'correct' device
    :return: 'correct' tensor
    """
    if tensor.device.type != device:
        tensor = tensor.to(device)
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor
