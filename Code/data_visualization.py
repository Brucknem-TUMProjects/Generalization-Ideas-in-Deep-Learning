import torchvision
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_samples(loader, classes=None, num_samples=8):
    # get some random training images
    dataiter = iter(loader)
    images, labels = dataiter.next()

    real_num_samples = min(num_samples, len(labels))
    images, labels = images[:real_num_samples], labels[:real_num_samples]

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    if classes:
        print('  ' + ' '.join('%8s' % classes[labels[j]] for j in range(real_num_samples)))
    else:
        print('  ' + ' '.join('%8s' % labels[j].item() for j in range(real_num_samples)))

