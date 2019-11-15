"""
Nice solver my Marcel Bruckner.
"""
import copy
from typing import Callable, Optional, List

import numpy as np
import torchvision

import data_loader
import data_visualization
import optimizers
import solver_visualization
from solver_load_save import *
from solver_visualization import *

HEADER = "[epoch, iteration] training loss | training accuracy"
ITERATION_FORMAT = '[%5d, %9d] %13.8f | %17.8f'
VALIDATION_FORMAT = len(HEADER) * "-" + "\n" + '[%5d, finished ] accuracy:       %17.8f'
HEADER += "\n" + len(HEADER) * "-"


class Solver:
    """ This class trains the given NN model. """

    def __init__(self,
                 model: torch.nn.Module,
                 strategy: dict = None,
                 data: dict = None,

                 best_model_parameters: dict = None,
                 best_solver: object = None,
                 best_validation_accuracy: float = 0.0,

                 training_loss_history: dict = None,
                 training_accuracy_history: dict = None,

                 epoch_training_accuracy_history: dict = None,
                 epoch_validation_accuracy_history: dict = None,
                 ):
        """
        Constructor

        :param model_type:
        :param model_parameters:
        :param best_model_parameters:
        :param best_solver:
        :param best_validation_accuracy:
        :param training_loss_history:
        :param training_accuracy_history:
        :param epoch_training_accuracy_history:
        :param epoch_validation_accuracy_history:
        :param log_buffer:
        """

        self.data = data
        self.strategy = strategy

        # Set up some variables for book-keeping
        self.model: torch.nn.Module = model

        self.best_model_parameters: dict = best_model_parameters
        self.best_solver: Optional[object] = best_solver
        self.best_validation_accuracy: float = best_validation_accuracy or 0.0

        self.training_loss_history: dict = training_loss_history or {0: 0}
        self.training_accuracy_history: dict = training_accuracy_history or {0: 0}

        self.epoch_training_accuracy_history: dict = epoch_training_accuracy_history or {0: 0}
        self.epoch_validation_accuracy_history: dict = epoch_validation_accuracy_history or {0: 0}

    def iterate(self, model: torch.nn.Module, trainings_loader: torch.utils.data.DataLoader,
                device: str, optimizer: Callable, criterion: Callable, total_epoch: int,
                log_every: int, plotter: solver_visualization.SolverPlotter, verbose: bool):
        """
        Iterates over all samples in the trainings loader and performs gradient descent based on
        the optimization strategy and the loss criterion

        :param total_epoch:
        :param model:
        :param trainings_loader:
        :param device:
        :param optimizer:
        :param criterion:
        :param log_every:
        :param plotter:
        :param verbose:
        :return:
        """
        running_loss, running_training_accuracy, epoch_running_training_accuracy = 0.0, 0.0, 0.0

        log_every = len(trainings_loader) if log_every >= len(trainings_loader) else log_every

        model.train()

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

                total_iteration = total_epoch * len(trainings_loader) + i + 1
                self.training_loss_history[total_iteration] = avg_loss
                self.training_accuracy_history[total_iteration] = avg_acc

                if plotter:
                    plotter.append_training_loss(total_iteration, avg_loss)
                    plotter.append_training_accuracy(total_iteration, avg_acc)

                running_loss, running_training_accuracy = 0.0, 0.0

                print_if_verbose(ITERATION_FORMAT % (total_epoch, i + 1, avg_loss, avg_acc), verbose)

        epoch_training_accuracy = epoch_running_training_accuracy / len(trainings_loader)

        return epoch_training_accuracy

    def validate(self, model: torch.nn.Module, validation_loader: torch.utils.data.DataLoader,
                 device: str, epoch: int, save_best: bool, folder: str, filename: str,
                 plotter: solver_visualization.SolverPlotter, verbose: bool):
        """
        Validates the model using the data provided by the validation loader

        :param model:
        :param validation_loader:
        :param device:
        :param epoch:
        :param save_best:
        :param folder:
        :param filename:
        :param plotter:
        :param verbose:
        :return:
        """
        total_validation_samples, correct_validation_samples = 0, 0

        model.eval()
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

        print_if_verbose(len(HEADER) * "-", verbose)
        print_if_verbose(VALIDATION_FORMAT % (epoch, val_accuracy), verbose)

        if plotter:
            plotter.append_epoch_validation_accuracy(epoch, val_accuracy)

    def train(self, training: dict = None, saving: dict = None):

        """
        Trains the network

        :param model:
        :param trainings_loader:
        :param validation_loader:
        :param strategy:
        :param training:
        :param saving:
        :return:
        """

        optimizer, criterion, optimizer_config, lr_decay = parse_strategy_settings(self.strategy or {})
        dataset, batch_size, subset_size, random_labels = parse_data_settings(self.data or {})

        epochs, log_every, validate, verbose, plot, save_on_training_100 = parse_training_settings(training or {})
        save_every_nth_epoch, save_best, save_latest, filename, folder = parse_save_settings(saving or {})

        if verbose:
            self.print_log()

        optimizer_func = getattr(optimizers, optimizer)
        criterion_func = getattr(optimizers, criterion)

        model = self.model
        device = initialize_model(model, verbose)

        data_loader_func = getattr(data_loader, dataset)
        trainings_loader = data_loader_func(True, batch_size, subset_size, random_labels)
        validation_loader = None
        if validate:
            validation_loader = data_loader_func(False, batch_size, subset_size, random_labels)

        previous_epochs = np.max(list(self.epoch_training_accuracy_history.keys()))

        plotter = SolverPlotter(self) if plot else None

        if verbose:
            helpers.print_separated("Starting training on model: %s" % model)
            helpers.print_separated("Epochs: %s, Logging every %s. iteration, Plotting: %s" %
                                    (epochs, log_every, plot))

        for epoch in range(epochs):  # loop over the dataset multiple times
            total_epoch = epoch + previous_epochs + 1

            print_if_verbose(HEADER, verbose)

            optimizer, next_lr = optimizer_func(model.parameters(), optimizer_config, lr_decay)
            optimizer_config['lr'] = next_lr
            self.strategy['optimizer_config'] = optimizer_config
            criterion = criterion_func()

            self.epoch_training_accuracy_history[total_epoch] = \
                self.iterate(model, trainings_loader, device, optimizer, criterion,
                             total_epoch, log_every, plotter, verbose)

            if plot:
                plotter.append_epoch_training_accuracy(total_epoch, self.epoch_training_accuracy_history[total_epoch])

            if validation_loader:
                self.validate(model, validation_loader, device, total_epoch,
                              save_best, folder, filename, plotter, verbose)

            print_if_verbose(verbose=verbose)

            if save_latest > 0 and total_epoch % save_latest == 0:
                self.save_solver(filename=filename, folder=folder, verbose=False)

            if save_every_nth_epoch > 0 and total_epoch % save_every_nth_epoch == 0:
                self.save_solver(filename=filename,
                                 folder=folder,
                                 epoch=total_epoch,
                                 verbose=False)
            # debug_sizes(self, total_epoch)

            if save_on_training_100 and (1 - self.epoch_training_accuracy_history[total_epoch]) < 1e-8:
                self.save_solver(filename=filename, folder=folder, epoch='_reached_100', verbose=verbose)
                break

        if save_latest:
            self.save_solver(filename=filename, folder=folder, verbose=False)

        model.cpu()
        del model
        torch.cuda.empty_cache()

        if verbose:
            helpers.print_separator()
            helpers.print_separated("Training finished.")

    def print_class_accuracies(self, classes: List[str] = None):
        """
        Prints the accuracies for every class. Uses the given names if specified

        :param classes:
        """
        print_class_accuracies(self, classes)

    def print_log(self):
        """
        Prints the whole log
        """

        print("Currently broken!")

        # for epoch in self.epoch_training_accuracy_history.keys():
        #     if epoch == 0:
        #         continue
        #     print(HEADER)
        #
        #     for iteration in self.training_loss_history.keys():
        #         if iteration == 0:
        #             continue
        #         print(ITERATION_FORMAT % (epoch, iteration, self.training_loss_history[iteration], self.training_accuracy_history[iteration]))
        #
        #     if epoch in self.epoch_validation_accuracy_history.keys():
        #         print(VALIDATION_FORMAT % (epoch, self.epoch_validation_accuracy_history[epoch]))
        #
        #         print()

    def print_bokeh_plots(self):
        """
        Prints the loss and accuracy plots

        """
        print_bokeh_plots(self)

    def save_best_solver(self, filename='solver_best.pth', folder='solvers', verbose=False):
        """
        Saves the best solver

        :param filename:
        :param folder:
        :param verbose:
        """
        save_best_solver(self, filename=filename, folder=folder, verbose=verbose)

    def save_solver(self, filename='solver.pth', folder='solvers', epoch='', verbose=False):
        """
        Saves the current solver

        :param filename:
        :param folder:
        :param epoch:
        :param verbose:
        :return:
        """
        save_solver(self, filename, folder, epoch, verbose)


def predict_samples(model, dataloader, classes=None, num_samples=8):
    """
    Picks some random samples from the validation data and predicts the labels.

    :param model:
    :param dataloader:
    :param classes:
    :param num_samples:
    :return:
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


def parse_data_settings(data: dict):
    """
    Parse the data settings dict

    :param data
    """
    dataset: str = data.get('dataset', 'cifar10')
    batch_size: int = data.get('batch_size', 16)
    subset_size: int = data.get('subset_size', -1)
    random_labels: bool = data.get('random_labels', False)

    return dataset, batch_size, subset_size, random_labels


def parse_strategy_settings(strategy: dict):
    """
    Parse the strategy settings dict

    :param strategy
    """
    optimizer: str = strategy.get('optimizer', 'adam')
    criterion: str = strategy.get('criterion', 'cross_entropy_loss')
    optimizer_config: dict = strategy.get('config', {})
    lr_decay: float = strategy.get('lr_decay', 0.9)

    return optimizer, criterion, optimizer_config, lr_decay


def parse_training_settings(training: dict):
    """
    Parse the training settings dict

    :param training
    """
    epochs: int = training.get('epochs', 10)
    verbose: bool = training.get('verbose', True)
    log_every: int = training.get('log_every', 50)
    plot: bool = training.get('plot', False)
    validate: bool = training.get('validate', True)
    save_on_training_100: bool = training.get('save_on_training_100', False)

    return epochs, log_every, validate, verbose, plot, save_on_training_100


def parse_save_settings(saving: dict):
    """
    Parse the load and save settings dict

    :param saving
    """
    nth_epoch: int = saving.get('nth_epoch', -1)
    best: bool = saving.get('best', True)
    latest: int = saving.get('latest', -1)
    filename: str = saving.get('filename', 'solver.pth')
    folder: str = saving.get('folder', 'solvers/')
    return nth_epoch, best, latest, filename, folder


def initialize_model(model, verbose=False):
    """
    Moves the model to GPU if possible. If given sets the model parameters

    :param model:
    :param verbose:
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        helpers.print_separated("Using device: %s" % device)

    torch.cuda.empty_cache()
    model.cuda(device)
    return device


def print_if_verbose(message="", verbose=False):
    """
    Appends the message to the log buffer. If verbose, then the message is also printed.

    :param message:
    :param verbose:
    """
    if verbose:
        print(message)

