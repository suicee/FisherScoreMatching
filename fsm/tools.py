
import numpy as np
import torch
import torch.nn as nn

def to_tensor(data):
    '''
    convert a numpy array to a torch tensor

    Parameters:
    data (numpy array): the data to be converted
    device (torch device): the device on which the data will be stored

    Returns:
    data (torch tensor): the converted data
    '''
    # if is tensor already return it
    if isinstance(data, torch.Tensor):
        return data
    # if is numpy array convert it to tensor
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()

def to_numpy(data):
    '''
    convert a torch tensor to a numpy array

    Parameters:
    data (torch tensor): the data to be converted

    Returns:
    data (numpy array): the converted data
    '''
    # if is tensor already return it
    if isinstance(data, np.ndarray):
        return data
    # if is numpy array convert it to tensor
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()##class for model training