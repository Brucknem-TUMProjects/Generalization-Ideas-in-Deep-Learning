import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import helpers

CIFAR10_DATASET_NAME = 'cifar10'

MNIST_DATASET_NAME = 'mnist'

CIFAR_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

MNIST_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def mnist(train, batch_size, subset_size, random_labels, confusion_size=0, verbose=True):
    """
    Loads the MNIST digit dataset

    :param confusion_size:
    :param verbose:
    :param train:
    :param batch_size:
    :param subset_size:
    :param random_labels:
    :return:
    """
    return get_dataloader(MNIST_DATASET_NAME,
                          train,
                          batch_size=batch_size,
                          subset_size=subset_size,
                          random_labels=random_labels,
                          confusion_size=confusion_size,
                          verbose=verbose)


def cifar10(train, batch_size, subset_size, random_labels, confusion_size=0, verbose=True):
    """
    Loads the CIFAR-10 dataset

    :param confusion_size:
    :param verbose:
    :param train:
    :param batch_size:
    :param subset_size:
    :param random_labels:
    :return:
    """
    return get_dataloader(CIFAR10_DATASET_NAME,
                          train,
                          batch_size=batch_size,
                          subset_size=subset_size,
                          random_labels=random_labels,
                          confusion_size=confusion_size,
                          verbose=verbose)


def load_dataset(dataset_name,
                 train,
                 download=True,
                 transform=None,
                 random_labels=False):
    """
    Loads the CIFAR-10 dataset

    :param dataset_name:
    :param train:
    :param download:
    :param transform:
    :param random_labels:
    :return:
    """
    dataset_settings = {
        'root': './' + dataset_name,
        'train': train,
        'download': download,
        'transform': transform if transform else
        CIFAR_DEFAULT_TRANSFORM if dataset_name == CIFAR10_DATASET_NAME else
        MNIST_DEFAULT_TRANSFORM if dataset_name == MNIST_DATASET_NAME else
        transforms.Compose([])
    }

    if dataset_name == CIFAR10_DATASET_NAME:
        dataset = torchvision.datasets.CIFAR10(**dataset_settings)
    elif dataset_name == MNIST_DATASET_NAME:
        dataset = torchvision.datasets.MNIST(**dataset_settings)

    np.random.seed(123456789)
    if random_labels:
        labels = list(dataset.class_to_idx.values())
        dataset.targets = np.random.choice(labels, len(dataset))

    return dataset


def get_dataloader(dataset_name,
                   train,
                   batch_size=64,
                   num_workers=8,
                   download=True,
                   transform=None,
                   subset_size=-1,
                   random_labels=False,
                   confusion_size=0,
                   verbose=True):
    """
    Loads the CIFAR-10 dataset and creates a data loader

    :param dataset_name:
    :param confusion_size:
    :param train:
    :param batch_size:
    :param num_workers:
    :param download:
    :param transform:
    :param subset_size:
    :param random_labels:
    :param verbose:
    :return:
    """
    if verbose:
        message = "Loading %s set: %s, Batch size: %s" % \
                  ('trainings' if train else 'validation', dataset_name, batch_size)
        message = message + ", Subset size: %s" % subset_size if subset_size >= 0 else message
        message = message + ", Confusion set size: %s" % confusion_size if confusion_size > 0 else message
        message = message + ", Random labels" if random_labels else message
        print(message)

    dataset = load_dataset(dataset_name,
                           train,
                           download=download,
                           transform=transform,
                           random_labels=random_labels)

    if subset_size < 0:
        subset_size = len(dataset)
    subset_size = min(subset_size + confusion_size, len(dataset))

    helpers.print_separator()

    random.seed(123456789)
    subset_indices = random.sample(range(len(dataset)), subset_size)

    if confusion_size:
        labels = list(dataset.class_to_idx.values())
        confusing_indices = subset_indices[-confusion_size:]
        targets = np.array(dataset.targets)
        targets[confusing_indices] = np.random.choice(labels, confusion_size)
        dataset.targets = list(targets)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices))


def get_CIFAR10_classes():
    """
    The CIFAR-10 class names

    :return:
    """
    return [
        'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
        'truck'
    ]
