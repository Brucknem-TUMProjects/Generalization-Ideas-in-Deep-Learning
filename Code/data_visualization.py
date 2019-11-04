import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Visualization

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_samples(trainloader, classes, num_samples=5):
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    real_num_samples = min(num_samples, len(labels))
    images, labels = images[:real_num_samples], labels[:real_num_samples]

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(real_num_samples)))
