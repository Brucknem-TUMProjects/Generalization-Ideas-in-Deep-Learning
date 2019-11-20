import concurrent
import copy
from collections import OrderedDict
from typing import List, Callable

import numpy as np
import torch

import data_loader


def margins(model: torch.nn.Module, training_loader: data_loader) -> List[float]:
    """
    f_{w}(x)[y_{true}] - max_{y != y_{true}}f_{w}[y]

    :param model:
    :param training_loader:
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    all_margins = []
    wrong_labelled = 0

    with torch.no_grad():
        for data in training_loader:
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

        # print(wrong_labelled)

    return all_margins


def gamma_margin(model: torch.nn.Module, training_loader: data_loader, eps: float = 0.01) -> float:
    """
    Lowest value of gamma so that ceil(eps * m) data points have a margin lower than gamma.
    m = #datapoints

    :param model:
    :param training_loader:
    :param eps:
    :return:
    """
    all_margins = sorted(margins(model, training_loader))
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


def l2_norm(model: torch.nn.Module, training_loader: data_loader, eps: float = 0.01) -> float:
    """
    Calculates the product of l2 norms over layers scaled by the gamma margin

    :param model:
    :param training_loader:
    :param eps:
    :return:
    """
    margin = gamma_margin(model, training_loader, eps)
    norm = 4 * norm_product(model.state_dict())

    return (margin ** -2) * norm


def spectral_norm(model: torch.nn.Module, training_loader: data_loader, eps: float = 0.01) -> float:
    """
    Calculates the product of spectral norms over layers scaled by the gamma margin

    :param model:
    :param training_loader:
    :param eps:
    :return:
    """
    margin = gamma_margin(model, training_loader, eps)
    norm = norm_product(model.state_dict(), order=2, with_hidden=True)

    return (margin ** -2) * norm


def l1_path_norm(model: torch.nn.Module, training_loader: data_loader, eps: float = 0.01) -> float:
    """
    Calculates the product of l1 path norms over layers scaled by the gamma margin

    :param model:
    :param training_loader:
    :param eps:
    :return:
    """
    margin = gamma_margin(model, training_loader, eps)

    layers = model.state_dict()
    paths = enumerate_paths(layers)
    paths *= 2
    paths = paths.prod(axis=1)
    paths = np.abs(paths)
    norm = paths.sum()
    norm **= 2

    return (margin ** -2) * norm


def l2_path_norm(model: torch.nn.Module, training_loader: data_loader, eps: float = 0.01) -> float:
    """
    Calculates the product of l2 path norms over layers scaled by the gamma margin

    :param model:
    :param training_loader:
    :param eps:
    :return:
    """
    margin = gamma_margin(model, training_loader, eps)

    layers = model.state_dict()

    paths = enumerate_paths(layers, power=2, multiply_by_hidden_units=True)
    paths *= 4
    paths = paths.prod(axis=1)
    norm = paths.sum()

    return (margin ** -2) * norm


def enumerate_paths(layers: OrderedDict, multiply_by_hidden_units: bool = False, power: int = 1):
    """
    Enumerates all paths in the given layer dict.

    :param layers:
    :param multiply_by_hidden_units:
    :param power:
    :return: List of list of weights representing all weights along paths
    """
    paths = np.array([])

    for layer in reversed(layers.keys()):
        if not layer.endswith('weight'):
            continue
        weights = layers[layer].cpu().numpy()
        rows = weights.shape[0]

        if not paths.any():
            paths = weights.flatten()
            paths **= power
            if multiply_by_hidden_units:
                paths *= rows
            paths = np.array([[i] for i in paths])
            continue

        reshaped_paths = [[] for x in np.arange(rows)]
        for i in range(len(paths)):
            reshaped_paths[i % rows].append(paths[i])
        paths = np.array(reshaped_paths)
        new_paths = []
        for i in range(len(paths)):
            row = paths[i]
            for element in row:
                for other_element in weights[i % weights.shape[0]]:
                    other_element **= power
                    if multiply_by_hidden_units:
                        other_element *= rows
                    new_paths.append(np.array([*element.flatten(), other_element]))

        paths = np.array(new_paths)
    return paths


def sharpness(model: torch.nn.Module, criterion: Callable, training_loader: data_loader, alpha: float = 5e-4,
              iterations: int = 1e5, num_threads: int = 8):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    real_loss = calculate_loss(model, criterion, training_loader)
    max_loss = -np.inf

    # alpha_interval = [-alpha, -alpha / 2.0, alpha / 2.0, alpha]

    model_parameters = OrderedDict(copy.deepcopy(dict(model.state_dict())))
    pertubated_parameters = OrderedDict()
    pertubation = OrderedDict()

    for iteration in range(iterations):
        for layer, weights in model_parameters.items():
            pertubation[layer] = ((np.random.rand(*weights.shape) * 2) - 1) * alpha

        for layer, weights in model_parameters.items():
            pertubated_parameters[layer] = model_parameters[layer] + torch.tensor(pertubation[layer]).to(device)

        model.load_state_dict(pertubated_parameters)
        loss = calculate_loss(model, criterion, training_loader)
        if loss > max_loss:
            max_loss = loss

    return max_loss

    # for layer in model.state_dict().keys():
    #     for row in range(len(model.state_dict()[layer])):
    #         if not len(model.state_dict()[layer][row].shape):
    #             pertubate_bias(model, criterion, training_loader, alpha_interval, real_loss, layer, row)
    #         else:
    #             for column in range(len(model.state_dict()[layer][row])):
    #                 print()


# def pertubate_bias(model: torch.nn.Module, criterion: Callable, training_loader: data_loader, alpha_interval: list,
#                    real_loss: float, layer: str, row: int) -> float:
#     """
#     Pertubates a bias layer and returns the max sharpness loss
#
#     :param layer:
#     :param real_loss:
#     :param row:
#     :param model:
#     :param criterion:
#     :param training_loader:
#     :param alpha_interval:
#     :return:
#     """
#     max_loss = -np.inf
#     original_value = model.state_dict()[layer][row].item()
#     for v in alpha_interval:
#         model.state_dict()[layer][row] = original_value + v
#         print(model.state_dict()[layer][row].item())
#         loss = calculate_loss(model, criterion, training_loader)
#
#         if loss > max_loss:
#             max_loss
#
#     model.state_dict()[layer][row] = original_value


def calculate_loss(model: torch.nn.Module, criterion: Callable, training_loader: data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        total_loss = 0.0

        for i, data in enumerate(training_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

        total_loss /= len(training_loader)

    return total_loss
