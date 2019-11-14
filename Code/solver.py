"""
Nice solver my Marcel Bruckner.
"""
import copy
import os


from typing import Callable

import numpy as np
import torch
import torchvision
from IPython import display

import data_visualization
import helpers
import optimizers
from solver_visualization import *
from solver_load_save import *

HEADER = "[epoch, iteration] training loss | training accuracy"


class Solver:
    """ This class trains the given NN model. """

    def __init__(self,
                 model_type: str = "",

                 model_parameters: dict = None,
                 best_model_parameters: dict = None,
                 best_solver: object = None,
                 best_validation_accuracy: float = 0.0,

                 training_loss_history: dict = None,
                 training_accuracy_history: dict = None,

                 epoch_training_accuracy_history: dict = None,
                 epoch_validation_accuracy_history: dict = None,

                 log_buffer: str = ""
                 ):
        """
        Constructor
        """

        # Set up some variables for book-keeping
        self.model_type = model_type

        self.model_parameters: dict = model_parameters
        self.best_model_parameters: dict = best_model_parameters
        self.best_solver: Solver = best_solver
        self.best_validation_accuracy: float = best_validation_accuracy or 0.0

        self.training_loss_history: dict = training_loss_history or {}
        self.training_accuracy_history: dict = training_accuracy_history or {}

        self.epoch_training_accuracy_history: dict = epoch_training_accuracy_history or {0: 0}
        self.epoch_validation_accuracy_history: dict = epoch_validation_accuracy_history or {0: 0}

        self.log_buffer: str = log_buffer or ""

    def set_model_type(self, model):
        if self.model_type:
            if type(model) is not self.model_type:
                raise ValueError(
                    'The model type is not equal to the previous used model type. Please use model of type % s'
                    % self.model_type
                )
        self.model_type = type(model)

    def iterate(self, model, trainings_loader, device, optimizer, criterion, epoch, log_every, plotter, verbose):
        running_loss, running_training_accuracy, epoch_running_training_accuracy = 0.0, 0.0, 0.0

        log_every = len(trainings_loader) - 1 if log_every >= len(trainings_loader) else log_every

        for i, data in enumerate(trainings_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted_train_labels = torch.max(outputs.data, 1)
            train_acc = (predicted_train_labels == labels).sum().item() / len(predicted_train_labels)

            # print statistics
            running_loss += loss.item()
            running_training_accuracy += train_acc
            epoch_running_training_accuracy += train_acc

            if i % log_every == (log_every - 1):
                avg_loss = running_loss / log_every
                avg_acc = running_training_accuracy / log_every

                total_iteration = (epoch - 1) * len(trainings_loader) + i
                self.training_loss_history[total_iteration] = avg_loss
                self.training_accuracy_history[total_iteration] = avg_acc

                if plotter:
                    plotter.append_training_loss(total_iteration, avg_loss)
                    plotter.append_training_accuracy(total_iteration, avg_acc)

                running_loss, running_training_accuracy, last_logged = 0.0, 0.0, 0.0

                self.print_and_buffer('[%5d, %9d] %13.8f | %17.8f' % (epoch, i + 1, avg_loss, avg_acc), verbose)

        self.model_parameters = model.state_dict()
        return epoch_running_training_accuracy / len(trainings_loader)

    def validate(self, model, validation_loader, device, epoch, save_best, folder, filename, plotter, verbose):
        total_validation_samples, correct_validation_samples = 0, 0

        with torch.no_grad():
            for data in validation_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                _, predicted_val_labels = torch.max(outputs.data, 1)
                total_validation_samples += labels.size(0)
                correct_validation_samples += (
                        predicted_val_labels == labels).sum().item()

        val_accuracy = correct_validation_samples / total_validation_samples
        self.epoch_validation_accuracy_history[epoch] = val_accuracy

        if val_accuracy > self.best_validation_accuracy:
            self.best_validation_accuracy = val_accuracy
            self.best_model_parameters = copy.deepcopy(model).cpu().state_dict()
            # self.best_model_parameters = model.state_dict()
            self.best_solver = copy.deepcopy(self)
            self.best_solver.best_solver = None

            if save_best:
                self.save_best_solver(folder=folder, filename=filename, verbose=False)

        self.print_and_buffer(len(HEADER) * "-", verbose)
        self.print_and_buffer('[%5d, %9s] %13s   %17.8f' % (epoch, "finished", "accuracy:", val_accuracy), verbose)

        if plotter:
            plotter.append_epoch_validation_accuracy(epoch, val_accuracy)

    def train(self,
              model: torch.nn.Module,
              trainings_loader: torch.utils.data.DataLoader = None,
              validation_loader: torch.utils.data.DataLoader = None,
              strategy: dict = None,
              training: dict = None,
              saving: dict = None
              ):

        """ Trains the network.
        Iterates over the trainings data num_epochs time and performs gradient descent
        based on the optimizer.

        model               -- the NN model to train (torch.nn.Module)
        trainings_loader    -- the trainings data (torch.utils.data.DataLoader)
        validation_loader    -- the validation data (torch.utils.data.DataLoader
        optimizer           -- the optimization strategy (sgd, adam, ...)
                                This is the name of one of the functions defined in optimizers.py
                                (default 'adam')
        criterion           -- the loss function criterion (l2, cross entropy loss, ...) -
                                This is the name of one of the functions defined in optimizers.py
                                (default 'cross_entropy_loss')
        optim_config        -- the configuration of the optimizer
            {
                    lr      -- the learning rate - (default 1.0)
            }
        lr_decay            -- the learning rate decay - (default 1.0)
        num_epochs          -- the number of epochs - (default 10)
        log_every           -- the number of iterations to pass between every verbose log - (default 100)
        verbose             -- wheather or not to log the trainings progress
        """

        if not trainings_loader:
            raise ValueError(
                'There is no trainings loader specified. Please set one via solver.trainings_loader = '
                '<torch.utils.data.DataLoader> '
            )

        self.set_model_type(model)

        optimizer_func, criterion_func, optimizer_config, lr_decay = parse_strategy_settings(strategy or {})
        epochs, log_every, verbose, plot = parse_training_settings(training or {})
        save_every_nth_epoch, save_best, save_latest, filename, folder = parse_save_settings(saving or {})

        device = initialize_model(model, self.model_parameters, verbose)

        previous_epochs = np.max(list(self.epoch_training_accuracy_history.keys()))

        plotter = SolverPlotter(self) if plot else None

        if verbose:
            helpers.print_separated(
                "Starting training.\n"
                "Epochs: %s, Logging every %s. iteration, Plotting: %s" % (epochs, log_every, plot)
            )

        for epoch in range(epochs):  # loop over the dataset multiple times
            total_epoch = epoch + previous_epochs + 1

            self.print_and_buffer(HEADER, verbose)
            self.print_and_buffer(len(HEADER) * "-", verbose)

            optimizer, next_lr = optimizer_func(model.parameters(), optimizer_config, lr_decay)
            optimizer_config['lr'] = next_lr
            criterion = criterion_func()

            self.epoch_training_accuracy_history[total_epoch] = \
                self.iterate(model, trainings_loader, device, optimizer, criterion,
                             total_epoch, log_every, plotter, verbose)

            if plot:
                plotter.append_epoch_training_accuracy(total_epoch, self.epoch_training_accuracy_history[-1])

            if validation_loader:
                self.validate(model, validation_loader, device, total_epoch,
                              save_best, folder, filename, plotter, verbose)

            self.print_and_buffer(verbose=verbose)

            if save_latest:
                self.save_solver(filename=filename, folder=folder, epoch='a' if total_epoch % 2 == 0 else 'b', verbose=False)

            if total_epoch % save_every_nth_epoch == 0:
                self.save_solver(filename=filename,
                                 folder=folder,
                                 epoch=total_epoch,
                                 verbose=False)
            # debug_sizes(self, total_epoch)

        model.cpu()
        del model
        torch.cuda.empty_cache()

        if verbose:
            helpers.print_separator()
            helpers.print_separated("Training finished.")

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

    def save_best_solver(self, filename='solver_best.pth', folder='solvers', verbose=False):
        save_best_solver(self, filename=filename, folder=folder, verbose=verbose)

    def save_solver(self, filename='solver.pth', folder='solvers', epoch='', verbose=False):
        save_solver(self, filename, folder, epoch, verbose)


def predict_samples(model, dataloader, classes=None, num_samples=8):
    """ Picks some random samples from the validation data and predicts the labels.
    classes -- list of classnames - (default=None)
    """

    # get some random training images
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    real_num_samples = min(num_samples, len(labels))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images, labels = images[:real_num_samples].to(device), labels[:real_num_samples].to(device)

    with torch.no_grad():
        outputs = model(images)
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


def parse_strategy_settings(strategy: dict):
    """

    :type strategy: dict
    """
    optimizer_func: Callable = getattr(optimizers, strategy.get('optimizer', 'adam'))
    criterion_func: Callable = getattr(optimizers, strategy.get('criterion', 'cross_entropy_loss'))
    optimizer_config: dict = strategy.get('config', {})
    lr_decay: float = strategy.get('lr_decay', 1.0)

    return optimizer_func, criterion_func, optimizer_config, lr_decay


def parse_training_settings(training: dict):
    """

    :type training: dict
    """
    epochs: int = training.get('epochs', 10)
    verbose: bool = training.get('verbose', True)
    log_every: int = training.get('log_every', 50)
    plot: bool = training.get('plot', False)

    return epochs, log_every, verbose, plot


def parse_save_settings(saving: dict):
    """

    :type saving: dict
    """
    nth_epoch: int = saving.get('nth_epoch', -1)
    best: bool = saving.get('best', True)
    latest: bool = saving.get('latest', True)
    filename: str = saving.get('filename', 'solver.pth')
    folder: str = saving.get('folder', 'solvers/')
    return nth_epoch, best, latest, filename, folder


def initialize_model(model, model_parameters=None, verbose=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        helpers.print_separated("Using device: %s" % device)

    if model_parameters:
        model.load_state_dict(model_parameters)

    torch.cuda.empty_cache()
    model.cuda(device)
    return device
