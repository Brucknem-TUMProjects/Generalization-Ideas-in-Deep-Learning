import torch.nn as nn
import torch.optim


def sgd(params, optim_params, lr_decay=0.9):
    """
    Stochastic gradient descent

    :param params:
    :param optim_params:
    :param lr_decay:
    :return:
    """
    """ Stocastic gradient descent. """
    lr = optim_params.get('lr', 0.001)
    momentum = optim_params.get('beta', 0.9)
    return torch.optim.SGD(params, lr=lr, momentum=momentum), lr * lr_decay


def adam(params, optim_params, lr_decay=0.9):
    """
    Adam

    :param params:
    :param optim_params:
    :param lr_decay:
    :return:
    """
    lr = optim_params.get('lr', 1e-3)
    betas = optim_params.get('betas', (0.9, 0.999))
    eps = optim_params.get('eps', 1e-8)
    weight_decay = optim_params.get('weight_decay', 0.0001)
    amsgrad = optim_params.get('amsgrad', False)

    return torch.optim.Adam(params, lr, betas, eps, weight_decay, amsgrad), lr * lr_decay


def cross_entropy_loss():
    """
    Cross entropy loss

    :return:
    """
    return nn.CrossEntropyLoss()
