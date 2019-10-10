# Helper functions
import torch
import numpy as np


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def de_variable(v):
    '''
    normalize the vector and detach it from variable
    '''

    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item() + 1e-6
    v = [vi / s for vi in v]
    return v


def get_blocks(model, blocktype):
    '''
    get the blocks in a model that has blocktype
    '''
    block_names = []
    blocks = []
    for name, module in model.named_modules():
        if isinstance(module, blocktype):
            block_names.append(name)
            blocks.append(module)
    return block_names, blocks


def get_max_error(eigenvalues, eigenvalues_old):
    '''
    the the maximum relative error between the eigenvalues list
    and the eigenvalues_old list
    '''
    eigenvalues = np.asarray(eigenvalues)
    eigenvalues_old = np.asarray(eigenvalues_old)
    relative_error = np.absolute((eigenvalues-eigenvalues_old)/eigenvalues)
    return np.amax(relative_error)
