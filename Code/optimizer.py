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

def sgd(params, optim_params):
    lr = optim_params.get('lr', 0.001)
    momentum = optim_params.get('beta', 0.9)
    return optim.SGD(params, lr=lr, momentum=momentum)


# Criteria

def cross_entropy_loss():
    return nn.CrossEntropyLoss()
