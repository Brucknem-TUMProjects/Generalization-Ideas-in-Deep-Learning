from pprint import pprint

import numpy as np


def margin():
    return 1


def l2_norm(layers):
    result = 1

    for layer, weights in layers.items():
        layer_result = weights**2
        layer_result = layer_result.sum()
        layer_result = layer_result**0.5
        layer_result *= 4

        result *= layer_result

    return result


def l1_path_norm(weights):
    pass


def l2_path_norm(weights):
    pass


def spectral_norm(weights):
    pass
