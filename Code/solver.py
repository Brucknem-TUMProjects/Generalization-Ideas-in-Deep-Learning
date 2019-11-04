import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import optimizer

class Solver:
    def __init__(self, model, trainloader, testloader, **kwargs):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = kwargs.pop('criterion', 'cross_entropy_loss')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_every = kwargs.pop('print_every', 100)
        self.verbose = kwargs.pop('verbose', True)

        if not hasattr(optimizer, self.criterion):
            raise ValueError('Invalid criterion "%s"' % self.criterion)
        self.criterion = getattr(optimizer, self.criterion)()

        self.optimizer = kwargs.pop('optimizer', 'sgd')
        if not hasattr(optimizer, self.optimizer):
            raise ValueError('Invalid optimizer "%s"' % self.optimizer)
        self.optimizer = getattr(optimizer, self.optimizer)(self.model.parameters(), self.optim_config)


        self._reset()


    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        # self.optim_configs = {}
        # for p in self.model.params:
            # d = {k: v for k, v in self.optim_config.items()}
            # self.optim_configs[p] = d


    def train(self):
        device = self.device

        for epoch in range(self.num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % self.print_every == (self.print_every - 1):    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / self.print_every))
                    running_loss = 0.0

        print('Finished Training')


    def save_model(self, path='./model.pth'):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='./model.pth'):
        self.model.load_state_dict(torch.load(path))




