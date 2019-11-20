from collections import OrderedDict

import numpy as np
import torch

import measures
import optimizers
from data_loader import get_test_data_loader
from networks import MultiLayerPerceptron
from solver import Solver

FC_PATHS = [
    [19, 7, 1],
    [19, 7, 2],
    [23, 7, 1],
    [23, 7, 2],
    [20, 10, 1],
    [20, 10, 2],
    [24, 10, 1],
    [24, 10, 2],
    [21, 13, 1],
    [21, 13, 2],
    [25, 13, 1],
    [25, 13, 2],
    [22, 16, 1],
    [22, 16, 2],
    [26, 16, 1],
    [26, 16, 2],
    [19, 8, 3],
    [19, 8, 4],
    [23, 8, 3],
    [23, 8, 4],
    [20, 11, 3],
    [20, 11, 4],
    [24, 11, 3],
    [24, 11, 4],
    [21, 14, 3],
    [21, 14, 4],
    [25, 14, 3],
    [25, 14, 4],
    [22, 17, 3],
    [22, 17, 4],
    [26, 17, 3],
    [26, 17, 4],
    [19, 9, 5],
    [19, 9, 6],
    [23, 9, 5],
    [23, 9, 6],
    [20, 12, 5],
    [20, 12, 6],
    [24, 12, 5],
    [24, 12, 6],
    [21, 15, 5],
    [21, 15, 6],
    [25, 15, 5],
    [25, 15, 6],
    [22, 18, 5],
    [22, 18, 6],
    [26, 18, 5],
    [26, 18, 6]]
FC_TEST_LAYERS = OrderedDict([
    ('layers.0.weight', torch.tensor([[1, 2], [3, 4], [5, 6]])),
    ('layers.0.bias', torch.tensor([0, 0, 0])),
    ('layers.1.weight', torch.tensor([[7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])),
    ('layers.1.bias', torch.tensor([0, 0, 0, 0])),
    ('layers.2.weight', torch.tensor([[19, 20, 21, 22], [23, 24, 25, 26]])),
    ('layers.2.bias', torch.tensor([0, 0]))
])
assert (np.array_equal(measures.enumerate_paths(FC_TEST_LAYERS), FC_PATHS))

net = MultiLayerPerceptron(**{'layers': [2, 3, 4, 2]})
net.load_state_dict(FC_TEST_LAYERS)
data_loader = get_test_data_loader()

assert(measures.gamma_margin(net, data_loader) == 1)

l1_path_norm = measures.l1_path_norm(net, data_loader)
assert(l1_path_norm == 155677593600.0)

l2_path_norm = measures.l2_path_norm(net, data_loader)
assert(l2_path_norm == 109977882624.0)

np.random.seed(123456789)
sharpness = measures.sharpness(net, optimizers.cross_entropy_loss(), data_loader, alpha=10, iterations=10)
assert(sharpness == 8081.87890625)
