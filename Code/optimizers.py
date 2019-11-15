import torch.nn as nn
import torch.optim


def sgd(params, optim_params, lr_decay=0.99):
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

    optim_params['lr'] = lr * lr_decay
    optim_params['momentum'] = momentum

    return torch.optim.SGD(params, lr=lr, momentum=momentum), optim_params


def adam(params, optim_params, lr_decay=0.99):
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

    optim_params['lr'] = lr * lr_decay
    optim_params['betas'] = betas
    optim_params['eps'] = eps
    optim_params['weight_decay'] = weight_decay
    optim_params['amsgrad'] = amsgrad

    return torch.optim.Adam(params, lr, betas, eps, weight_decay, amsgrad), optim_params


def cross_entropy_loss():
    """
    Cross entropy loss

    :return:
    """
    return nn.CrossEntropyLoss()
