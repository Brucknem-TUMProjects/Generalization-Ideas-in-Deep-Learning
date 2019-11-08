import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import optimizer
import data_visualization

class Solver:
    """ This class trains the given NN model. """

    def __init__(self, model, trainloader, validationloader, optim='adam', criterion='cross_entropy_loss', optim_config={},
                lr_decay=1.0, num_epochs=10, verbose=True, print_every=100):
        """
        Constructor

        model               -- the NN model to train (nn.Module)
        trainloader         -- the trainings data (torch.utils.data.DataLoader)
        validationloader    -- the validation data (torch.utils.data.DataLoader
        optimizer           -- the optimization strategy (sgd, adam, ...) - This is the name of one of the functions defined in optimizer.py - (default 'adam')
        criterion           -- the loss function criterion (l2, cross entropy loss, ...) - This is the name of one of the functions defined in optimizer.py - (default 'cross_entropy_loss')
        optim_config        -- the configuration of the optimizer
            {
                    lr      -- the learning rate - (default 1.0)
            }
        lr_decay            -- the learning rate decay - (default 1.0)
        num_epochs          -- the number of epochs - (default 10)
        print_every         -- the number of iterations to pass between every verbose log - (default 100)
        verbose             -- wheather or not to log the trainings progress

        """
        self.model = model
        self.trainloader = trainloader
        self.validationloader = validationloader

        """ The device used for calculation. CUDA or CPU. """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optim
        self.criterion = criterion
        self.optim_config = optim_config
        self.lr_decay = lr_decay
        self.num_epochs = num_epochs

        self.verbose = verbose
        self.print_every = print_every

        if not hasattr(optimizer, self.criterion):
            raise ValueError('Invalid criterion "%s"' % self.criterion)
        self.criterion = getattr(optimizer, self.criterion)

        if not hasattr(optimizer, self.optimizer):
            raise ValueError('Invalid optimizer "%s"' % self.optimizer)
        self.optimizer = getattr(optimizer, self.optimizer)


        self._reset()


    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
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
        """ Trains the network.
        Iterates over the trainings data num_epochs time and performs gradient descent
        based on the optimizer.
        """

        device = self.device
        header = "[epoch, iteration] training loss | training accuracy"

        optim_config = self.optim_config

        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            if self.verbose:
                print(header)
                print(len(header) * "-")

            optimizer, next_lr = self.optimizer(self.model.parameters(), optim_config)
            optim_config['lr'] = next_lr
            criterion = self.criterion()

            running_loss = 0.0
            running_training_accuracy = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                self.loss_history.append(loss.item())
                _, predicted_train_labels = torch.max(outputs.data, 1)
                train_acc = (predicted_train_labels == labels).sum().item() / len(predicted_train_labels)
                self.train_acc_history.append(train_acc)

                # print statistics
                running_loss += loss.item()
                running_training_accuracy += train_acc
                if self.verbose and i % self.print_every == (self.print_every - 1):
                    print('[%5d, %9d] %13.8f | %17.8f' %
                          (epoch + 1, i + 1, running_loss / self.print_every, running_training_accuracy / self.print_every))
                    running_loss = 0.0
                    running_training_accuracy = 0.0

            # Validation stuff
            total_validation_samples = 0
            correct_validation_samples = 0
            with torch.no_grad():
                for data in self.validationloader:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = self.model(inputs)
                    _, predicted_val_labels = torch.max(outputs.data, 1)
                    total_validation_samples += labels.size(0)
                    correct_validation_samples += (predicted_val_labels == labels).sum().item()

            val_accuracy = correct_validation_samples / total_validation_samples
            if self.verbose:
                print(len(header) * "-")
                print('[%5d, %9s] %13s | %17.8f \n' %
                      (epoch + 1, "finished", "accuracy:" , val_accuracy))



    def save_model(self, filename='model.pth'):
        """ Saves the model parameters.
        filename -- The file to which the parameters get saved. - default ('model.pth')
        """
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename='model.pth'):
        """ Loads the model parameters.
        filename -- The file from which the parameters get loaded. - default ('model.pth')
        """
        self.model.load_state_dict(torch.load(filename))

    def print_class_accuracies(self, classes=None):
        """ Prints the class accuracies.
        classes -- list of classnames - (default=None)
        """
        device = self.device
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for data in self.validationloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i].item()

                    class_correct.setdefault(label, 0)
                    class_correct[label] += c[i].item()
                    class_total.setdefault(label, 0)
                    class_total[label] += 1

        if not classes:
            classes = list(class_total.keys())

        for i in range(len(classes)):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))


    # Visualization
    def predict_samples(self, classes=None, num_samples=8):
        device = self.device

        # get some random training images
        dataiter = iter(self.trainloader)
        images, labels = dataiter.next()

        real_num_samples = min(num_samples, len(labels))
        images, labels = images[:real_num_samples].to(device), labels[:real_num_samples].to(device) 

        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)

            # show images
            data_visualization.imshow(torchvision.utils.make_grid(images.cpu()))
            if classes:
                print('%10s: %s' % ('Real', ' '.join('%5s' % classes[labels[j]] for j in range(real_num_samples))))
                print('%10s: %s' % ('Predicted', ' '.join('%5s' % classes[predicted[j]] for j in range(real_num_samples))))
            else:
                print('%10s: %s' % ('Real', ' '.join('%5s' % labels[j].item() for j in range(real_num_samples))))
                print('%10s: %s' % ('Predicted', ' '.join('%5s' % predicted[j].item() for j in range(real_num_samples))))

