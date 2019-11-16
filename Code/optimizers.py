import torch.nn as nn
import torch.optim


def sgd(params, optim_params, state=None):
    """
    Stochastic gradient descent

    :param state:
    :param params:
    :param optim_params:
    :return:
    """
    """ Stocastic gradient descent. """
    lr = optim_params.get('lr', 0.001)
    momentum = optim_params.get('beta', 0.9)

    optim_params['lr'] = lr
    optim_params['momentum'] = momentum

    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    if state:
        optimizer.load_state_dict(state)
    return optimizer, optim_params


def adam(params, optim_params, state=None):
    """
    Adam

    :param state:
    :param params:
    :param optim_params:
    :return:
    """
    lr = optim_params.get('lr', 1e-3)
    betas = optim_params.get('betas', (0.9, 0.999))
    eps = optim_params.get('eps', 1e-8)
    weight_decay = optim_params.get('weight_decay', 0.0001)
    amsgrad = optim_params.get('amsgrad', False)

    optim_params['lr'] = lr
    optim_params['betas'] = betas
    optim_params['eps'] = eps
    optim_params['weight_decay'] = weight_decay
    optim_params['amsgrad'] = amsgrad

    optimizer = torch.optim.Adam(params, lr, betas, eps, weight_decay, amsgrad)
    if state:
        optimizer.load_state_dict(state)
    return optimizer, optim_params


def cross_entropy_loss():
    """
    Cross entropy loss

    :return:
    """
    return nn.CrossEntropyLoss()
