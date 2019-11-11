import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

_default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def cifar10(train, batch_size, subset_size, random_labels):

    if train:
        return get_CIFAR10_trainset_loader(batch_size=batch_size,
                                           subset_size=subset_size,
                                           random_labels=random_labels)

    return get_CIFAR10_testset_loader(batch_size=batch_size,
                                      subset_size=subset_size,
                                      random_labels=random_labels)


def load_CIFAR10_dataset(train,
                         download=True,
                         transform=_default_transform,
                         subset_size=-1,
                         random_labels=False):
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=train,
                                            download=download,
                                            transform=_default_transform)

    if subset_size <= 0:
        subset_size = len(trainset)

    if random_labels:
        labels = list(trainset.class_to_idx.values())
        trainset.targets = np.random.choice(labels, len(trainset))

    return trainset


def get_CIFAR10_trainset_loader(batch_size=64,
                                num_workers=8,
                                download=True,
                                transform=_default_transform,
                                subset_size=-1,
                                random_labels=False):
    trainset = load_CIFAR10_dataset(True,
                                    download=download,
                                    transform=transform,
                                    subset_size=subset_size,
                                    random_labels=random_labels)

    if subset_size <= 0:
        subset_size = len(trainset)

    return torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            random.sample(range(0, len(trainset)), subset_size)))


def get_CIFAR10_testset_loader(batch_size=64,
                               num_workers=8,
                               download=True,
                               transform=_default_transform,
                               subset_size=-1,
                               random_labels=False):
    trainset = load_CIFAR10_dataset(False,
                                    download=download,
                                    transform=transform,
                                    subset_size=subset_size,
                                    random_labels=random_labels)

    if subset_size <= 0:
        subset_size = len(trainset)

    return torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            random.sample(range(0, len(trainset)), subset_size)))


def get_CIFAR10_classes():
    return [
        'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
        'truck'
    ]
