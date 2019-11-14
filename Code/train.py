import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pprint import pprint

from networks.networks import ExampleNet, vgg11, extract_layer_names
from solver import Solver, load_solver
import data_visualization
import data_loader


batch_size = 64
subset_size = 1000
random_labels = False
#trainloader = data_loader.get_CIFAR10_trainset_loader(batch_size=4, subset_indices=subset_indices)

trainloader = data_loader.get_CIFAR10_trainset_loader(batch_size=batch_size, subset_size=subset_size, random_labels=random_labels)
testloader = data_loader.get_CIFAR10_testset_loader(batch_size=batch_size, subset_size=subset_size)

classes = data_loader.get_CIFAR10_classes()


net = ExampleNet()
# net = models.vgg16(pretrained=False)
# net = vgg11()
# testloader=None
solver = Solver(net, trainings_loader=trainloader, validation_loader=testloader, strategy='adam')

solver.train(num_epochs=5, log_every=4, plot=False, verbose=True)

solver.save_solver()
solver.save_best_solver()


