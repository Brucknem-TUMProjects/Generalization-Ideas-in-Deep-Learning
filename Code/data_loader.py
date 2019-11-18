import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import helpers

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


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
    if train:
        return get_CIFAR10_dataloader(train,
                                      batch_size=batch_size,
                                      subset_size=subset_size,
                                      random_labels=random_labels,
                                      confusion_size=confusion_size,
                                      verbose=verbose)

    return get_CIFAR10_dataloader(train,
                                  batch_size=batch_size,
                                  subset_size=subset_size,
                                  random_labels=random_labels,
                                  confusion_size=confusion_size,
                                  verbose=verbose)


def load_CIFAR10_dataset(train,
                         download=True,
                         transform=DEFAULT_TRANSFORM,
                         random_labels=False):
    """
    Loads the CIFAR-10 dataset

    :param train:
    :param download:
    :param transform:
    :param random_labels:
    :return:
    """
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=train,
                                            download=download,
                                            transform=transform)

    np.random.seed(123456789)
    if random_labels:
        labels = list(trainset.class_to_idx.values())
        trainset.targets = np.random.choice(labels, len(trainset))

    return trainset


def get_CIFAR10_dataloader(train,
                           batch_size=64,
                           num_workers=8,
                           download=True,
                           transform=DEFAULT_TRANSFORM,
                           subset_size=-1,
                           random_labels=False,
                           confusion_size=0,
                           verbose=True):
    """
    Loads the CIFAR-10 dataset and creates a data loader

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
        message = "Loading %s set: %s, Batch size: %s" % ('trainings' if train else 'validation', 'cifar10', batch_size)
        message = message + ", Subset size: %s" % subset_size if subset_size >= 0 else message
        message = message + ", Confusion set size: %s" % confusion_size if confusion_size > 0 else message
        message = message + ", Random labels" if random_labels else message
        print(message)

    trainset = load_CIFAR10_dataset(train,
                                    download=download,
                                    transform=transform,
                                    random_labels=random_labels)

    if subset_size < 0:
        subset_size = len(trainset)
    subset_size = min(subset_size + confusion_size, len(trainset))

    helpers.print_separator()

    random.seed(123456789)
    subset_indices = random.sample(range(len(trainset)), subset_size)

    if confusion_size:
        labels = list(trainset.class_to_idx.values())
        confusing_indices = subset_indices[-confusion_size:]
        targets = np.array(trainset.targets)
        targets[confusing_indices] = np.random.choice(labels, confusion_size)
        trainset.targets = list(targets)

    return torch.utils.data.DataLoader(
        trainset,
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
