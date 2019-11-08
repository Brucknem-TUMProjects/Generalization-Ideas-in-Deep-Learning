import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Update rules

def sgd(params, optim_params, lr_decay=1.0):
    """ Stocastic gradient descent. """
    lr = optim_params.get('lr', 0.001)
    momentum = optim_params.get('beta', 0.9)
    return optim.SGD(params, lr=lr, momentum=momentum), lr * lr_decay

def adam(params, optim_params, lr_decay=1.0):
    """ Adam solver. """
    lr = optim_params.get('lr', 1e-3)
    betas = optim_params.get('betas', (0.9, 0.999))
    eps = optim_params.get('eps', 1e-8)
    weight_decay = optim_params.get('weight_decay', 0)
    amsgrad = optim_params.get('amsgrad', False)

    return optim.Adam(params, lr, betas, eps, weight_decay, amsgrad), lr * lr_decay

# Criteria

def cross_entropy_loss():
    return nn.CrossEntropyLoss()
