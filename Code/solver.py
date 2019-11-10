"""
Nice solver my Marcel Bruckner.
"""
import os

import _pickle as cPickle
import numpy as np
import torch
import torchvision
from IPython import display

import data_visualization
import optimizer
from solver_visualization import *


class Solver:
    """ This class trains the given NN model. """
    def __init__(self,
                 model,
                 trainloader,
                 validationloader=None,
                 optim='adam',
                 criterion='cross_entropy_loss',
                 optim_config={},
                 lr_decay=1.0,
                 best_val_acc=0,
                 val_acc_history={},
                 loss_history={},
                 per_iteration_train_acc_history={},
                 per_epoch_train_acc_history={},
                 log_buffer=""):
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
        log_every         -- the number of iterations to pass between every verbose log - (default 100)
        verbose             -- wheather or not to log the trainings progress

        """
        self.model = model
        self.trainloader = trainloader
        self.validationloader = validationloader
        """ The device used for calculation. CUDA or CPU. """
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optim = optim
        self.criterion = criterion
        self.optim_config = optim_config
        self.lr_decay = lr_decay

        if not hasattr(optimizer, self.criterion):
            raise ValueError('Invalid criterion "%s"' % self.criterion)
        self.criterion = getattr(optimizer, self.criterion)

        if not hasattr(optimizer, self.optim):
            raise ValueError('Invalid optimizer "%s"' % self.optim)
        self.optim = getattr(optimizer, self.optim)

        # Set up some variables for book-keeping

        self.best_val_acc = best_val_acc
        self.best_params = self.model.parameters()
        self.val_acc_history = val_acc_history

        self.loss_history = loss_history
        self.per_iteration_train_acc_history = per_iteration_train_acc_history
        self.per_epoch_train_acc_history = per_epoch_train_acc_history

        self.log_buffer = log_buffer

    def train(self,
              optim_config={},
              lr_decay=0,
              num_epochs=10,
              verbose=True,
              log_every=100,
              plot=False):
        """ Trains the network.
        Iterates over the trainings data num_epochs time and performs gradient descent
        based on the optimizer.
        """

        if not self.val_acc_history:
            self.val_acc_history[0] = 0

        if not self.per_epoch_train_acc_history:
            self.per_epoch_train_acc_history[0] = 0

        if not optim_config:
            optim_config = self.optim_config

        if not lr_decay:
            lr_decay = self.lr_decay

        log_every = min(log_every, len(self.trainloader) - 1)

        if plot:
            plotter = SolverPlotter(self)

        device = self.device
        header = "[epoch, iteration] training loss | training accuracy"

        previous_epochs = np.max(list(self.per_epoch_train_acc_history.keys()))

        for epoch in range(num_epochs):  # loop over the dataset multiple times

            total_epoch = epoch + previous_epochs + 1

            self.print_and_buffer(header, verbose)
            self.print_and_buffer(len(header) * "-", verbose)

            optimizer, next_lr = self.optim(self.model.parameters(),
                                            optim_config)
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

                _, predicted_train_labels = torch.max(outputs.data, 1)
                train_acc = (predicted_train_labels == labels
                             ).sum().item() / len(predicted_train_labels)

                # print statistics
                running_loss += loss.item()
                running_training_accuracy += train_acc

                if i % log_every == (log_every - 1):
                    avg_loss = running_loss / log_every
                    avg_acc = running_training_accuracy / log_every

                    total_iteration = (total_epoch - 1) * \
                        len(self.trainloader) + i
                    self.loss_history[total_iteration] = avg_loss
                    self.per_iteration_train_acc_history[
                        total_iteration] = avg_acc

                    if plot:
                        plotter.append_training_loss(total_iteration, avg_loss)
                        plotter.append_training_accuracy(
                            total_iteration, avg_acc)

                    running_loss = 0.0
                    running_training_accuracy = 0.0

                    self.print_and_buffer(
                        '[%5d, %9d] %13.8f | %17.8f' %
                        (total_epoch, i + 1, avg_loss, avg_acc), verbose)

            self.per_epoch_train_acc_history[total_epoch] = avg_acc

            if plot:
                plotter.append_epoch_training_accuracy(total_epoch, avg_acc)

            if self.validationloader:
                # Validation stuff
                total_validation_samples = 0
                correct_validation_samples = 0
                with torch.no_grad():
                    for data in self.validationloader:
                        inputs, labels = data[0].to(device), data[1].to(device)
                        outputs = self.model(inputs)
                        _, predicted_val_labels = torch.max(outputs.data, 1)
                        total_validation_samples += labels.size(0)
                        correct_validation_samples += (
                            predicted_val_labels == labels).sum().item()

                val_accuracy = correct_validation_samples / total_validation_samples
                self.val_acc_history[total_epoch] = val_accuracy

                if val_accuracy > self.best_val_acc:
                    self.best_val_acc = val_accuracy
                    self.best_params = self.model.state_dict()

                    for k, v in self.best_params.items():
                        self.best_params[k] = v.cpu()

                self.print_and_buffer(len(header) * "-", verbose)
                self.print_and_buffer(
                    '[%5d, %9s] %13s | %17.8f' %
                    (total_epoch, "finished", "accuracy:", val_accuracy),
                    verbose)

                if plot:
                    plotter.append_epoch_validation_accuracy(
                        total_epoch, val_accuracy)

            self.print_and_buffer(verbose=verbose)

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
        print_class_accuracies(self, classes)

    def print_log(self):
        print(self.log_buffer)

    def print_bokeh_plots(self):
        print_bokeh_plots(self)

    def print_and_buffer(self, message="", verbose=False):
        self.log_buffer += message + "\n"

        if verbose:
            print(message)

    def predict_samples(self, classes=None, num_samples=8):
        """ Picks some random samples from the validation data and predicts the labels.
        classes -- list of classnames - (default=None)
        """
        device = self.device

        # get some random training images
        dataiter = iter(self.trainloader)
        images, labels = dataiter.next()

        real_num_samples = min(num_samples, len(labels))
        images, labels = images[:real_num_samples].to(
            device), labels[:real_num_samples].to(device)

        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)

            # show images
            data_visualization.imshow(torchvision.utils.make_grid(
                images.cpu()))

            if classes:
                print('%10s: %s' %
                      ('Real', ' '.join('%8s' % classes[labels[j]]
                                        for j in range(real_num_samples))))
                print(
                    '%10s: %s' %
                    ('Predicted', ' '.join('%8s' % classes[predicted[j]]
                                           for j in range(real_num_samples))))
            else:
                print('%10s: %s' %
                      ('Real', ' '.join('%8s' % labels[j].item()
                                        for j in range(real_num_samples))))
                print(
                    '%10s: %s' %
                    ('Predicted', ' '.join('%8s' % predicted[j].item()
                                           for j in range(real_num_samples))))

    def save_solver(self, filename='history.pth', dir='solvers'):
        output_dict = dict(self.__dict__)

        del output_dict['device']
        del output_dict['best_params']

        output_dict['model'].cpu()
        output_dict['criterion'] = output_dict['criterion'].__name__
        output_dict['optim'] = output_dict['optim'].__name__

        if not os.path.exists(dir):
            os.makedirs(dir)
        cPickle.dump(output_dict, open(dir + '/' + filename, 'wb'), 2)


def load_solver(filename='history.pth', dir='solvers'):
    data = cPickle.load(open(dir + '/' + filename, 'rb'))

    return Solver(**data)
