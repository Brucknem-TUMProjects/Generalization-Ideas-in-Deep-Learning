from collections import OrderedDict
from typing import List

import numpy as np
import torch

import data_loader


def margins(model: torch.nn.Module, trainings_loader: data_loader) -> List[float]:
    """
    f_{w}(x)[y_{true}] - max_{y != y_{true}}f_{w}[y]

    :param model:
    :param trainings_loader:
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    all_margins = []
    wrong_labelled = 0

    with torch.no_grad():
        for data in trainings_loader:
            inputs, labels = data[0].cuda(device), data[1].numpy()
            outputs = model(inputs).cpu().numpy()
            correct_labels = []

            for i in range(len(labels)):
                outputs[i] -= np.min(outputs[i])
                outputs[i] /= np.sum(outputs[i])
                predicted = np.argmax(outputs[i])
                correct = labels[i]
                if predicted != correct:
                    wrong_labelled += 1
                correct_labels.append(outputs[i, labels[i]])
                outputs[i, labels[i]] = -np.inf
                max_other = np.amax(outputs[i])
                margin = correct_labels - max_other
                all_margins.extend(margin)

        print(wrong_labelled)

    return all_margins


def gamma_margin(model: torch.nn.Module, trainings_loader: data_loader, eps: float) -> float:
    """
    Lowest value of gamma so that ceil(eps * m) data points have a margin lower than gamma.
    m = #datapoints

    :param model:
    :param trainings_loader:
    :param eps:
    :return:
    """
    all_margins = sorted(margins(model, trainings_loader))
    if all_margins[0] < 0:
        print(all_margins[:10])
        raise ValueError("Found wrong labeled data.")

    eps_m = np.math.ceil(eps * len(all_margins))
    return all_margins[eps_m]


def norm_product(layers: OrderedDict, order: int = None, with_hidden: bool = False) -> float:
    """
    Calculates the product of norms over layers.

    :param with_hidden:
    :param layers:
    :param order:
    :return:
    """
    result = 1

    for layer, weights in layers.items():
        if isinstance(weights, torch.Tensor):
            weights = weights.cpu().numpy()

        layer_result = 1
        try:
            print("Processing layer %s" % layer)
            if layer.endswith('weight'):
                layer_result = np.linalg.norm(weights, ord=order)
                if with_hidden:
                    layer_result *= weights.shape[0]
            else:
                print("Skipping %s" % layer)
        except ValueError:
            print("Layer %s has no norm. %s" % (layer, weights.shape))

        result *= layer_result

    return result


def l2_norm(model: torch.nn.Module, trainings_loader: data_loader, eps: float = 0.01) -> float:
    """
    Calculates the product of l2 norms over layers scaled by the gamma margin

    :param model:
    :param trainings_loader:
    :param eps:
    :return:
    """
    margin = gamma_margin(model, trainings_loader, eps)
    norm = 4 * norm_product(model.state_dict())

    return (margin ** -2) * norm


def spectral_norm(model: torch.nn.Module, trainings_loader: data_loader, eps: float = 0.01) -> float:
    """
    Calculates the product of spectral norms over layers scaled by the gamma margin

    :param model:
    :param trainings_loader:
    :param eps:
    :return:
    """
    margin = gamma_margin(model, trainings_loader, eps)
    norm = norm_product(model.state_dict(), order=2, with_hidden=True)

    return (margin ** -2) * norm


def l1_path_norm(weights):
    return 0


def l2_path_norm(weights):
    return 0
