import torch
import torchvision
import torchvision.transforms as transforms


_default_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_CIFAR10_trainset_loader(batch_size=100, num_workers=8):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=_default_transform)
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

def get_CIFAR10_testset_loader(batch_size=100, num_workers=8):
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=_default_transform)
    return torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)

def get_CIFAR10_classes():
    return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
