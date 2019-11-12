from pprint import pprint

import numpy as np
import torch

def margin():
    return 1


def l2_norm_product(layers):
    result = 1

    for layer, weights in layers.items():
        if isinstance(weights, torch.Tensor):
            weights = weights.numpy()

        layer_result = np.linalg.norm(weights)

        result *= layer_result

    return result


def l1_path_norm(weights):
    pass


def l2_path_norm(weights):
    pass


def spectral_norm_product(layers):
    result = 1

    for layer, weights in layers.items():
        if isinstance(weights, torch.Tensor):
            weights = weights.numpy()
        layer_result = np.matmul(weights, weights.transpose())
        layer_result = np.linalg.eig(layer_result)[0]
        layer_result = layer_result.max()
        layer_result = layer_result**0.5

        result *= layer_result

    return result