from pprint import pprint

import numpy as np


def margin():
    return 1


def lpq_norm(matrix, p, q):
    if p is np.inf and p is np.inf:
        return np.max(matrix)

    result = np.absolute(matrix)
    result = result**p
    result = result.sum(axis=1)
    result = result**(q / p)
    result = result.sum()
    result = result**(1 / q)

    return result


def l2_norm_product(layers):
    result = 1

    for layer, weights in layers.items():
        layer_result = lpq_norm(weights, 2, 2)

        result *= layer_result

    return result


def l1_path_norm(weights):
    pass


def l2_path_norm(weights):
    pass


def spectral_norm_product(layers):
    result = 1

    for layer, weights in layers.items():
        layer_result = np.matmul(weights, weights.transpose())
        layer_result = np.linalg.eig(layer_result)[0]
        layer_result = layer_result.max()
        layer_result = layer_result**0.5

        result *= layer_result

    return result
