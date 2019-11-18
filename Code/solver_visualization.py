import math

import bokeh
import matplotlib.pyplot as plt
import torch
from bokeh.io import output_notebook, push_notebook, show
from bokeh.layouts import row
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

import helpers

BOKEH_VERSION = bokeh.__version__


def debug_sizes(solver, total_epoch=0):
    """
    Gets a rough estimate of the solvers size on drive

    :param solver:
    :param total_epoch:
    :return:
    """
    print("%34s: %15s" %
          ("Epoch " + str(total_epoch), helpers.get_size(solver)))

    for k, v in solver.__dict__.items():
        print("%34s: %15d" % (k, helpers.get_size(v)))

    print(80 * "*")


def print_class_accuracies(solver, classes=None):
    """
    Prints the class accuracies.

    :param solver:
    :param classes:
    :return:
    """

    if not solver.validationloader:
        print("Couldn't calculate accuracies as no validation set was given.")

        return

    device = solver.device
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for data in solver.validationloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = solver.model(inputs)
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
        print('Accuracy of %5s : %2d %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))


def print_plots(solver):
    """
    Prints the plots for
        - Training loss
        - Per iteration Training accuracy
        - Validation accuracy
        - Per epoch Training accuracy

    :param solver:
    :return:
    """

    plt.subplot(2, 1, 1)
    plt.title('Training')

    plt.plot(solver.loss_history, 'o', label='loss')
    plt.plot(solver.per_iteration_train_acc_history, 'o', label='accuracy')
    plt.xlabel('Iteration')
    plt.legend(loc='lower right')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(solver.per_epoch_train_acc_history, '-o', label='train')

    if solver.validationloader:
        plt.plot(solver.val_acc_history, '-o', label='val')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()


class SolverPlotter:
    """
    Class to plot the solver histories
    """

    def __init__(self, solver):
        """
        Constructor

        :param solver:
        """
        output_notebook()
        self.tools = "pan,wheel_zoom,box_zoom,reset,save,crosshair, hover"

        upper_row_settings = {
            'plot_width': 450,
            'plot_height': 400,
            'tools': self.tools,
            'x_axis_label': 'Iteration'
        }

        loss_plt = figure(**upper_row_settings,
                          title='Training Loss')

        iterations_per_epoch = math.ceil(solver.data['subset_size'] / solver.data['batch_size'])

        flattened_training_x_labels = []
        loss_history = []
        for epoch, iterations in solver.training_loss_history.items():
            start_iteration = epoch * iterations_per_epoch
            loss_history.extend(iterations.values())
            for iteration in iterations.keys():
                flattened_training_x_labels += [start_iteration + iteration]

        loss_plt_data = ColumnDataSource(
            data=dict(x=flattened_training_x_labels,
                      y=loss_history))
        loss_plt.line('x',
                      'y',
                      source=loss_plt_data,
                      line_color='red',
                      line_width=1)

        flattened_training_x_labels = []
        accuracy_history = []
        for epoch, iterations in solver.training_accuracy_history.items():
            start_iteration = epoch * iterations_per_epoch
            accuracy_history.extend(iterations.values())
            for iteration in iterations.keys():
                flattened_training_x_labels += [start_iteration + iteration]
        accuracy_history = [a if a <= 1.0 else 1.0 for a in accuracy_history]

        accuracy_plt = figure(**upper_row_settings,
                              title='Training Accuracy')
        accuracy_plt_data = ColumnDataSource(
            data=dict(x=flattened_training_x_labels,
                      y=accuracy_history))
        accuracy_plt.line('x',
                          'y',
                          source=accuracy_plt_data,
                          line_color='blue',
                          line_width=1)

        training_handle = show(row(loss_plt, accuracy_plt),
                               notebook_handle=True)

        epoch_plot = figure(plot_width=900,
                            plot_height=400,
                            tools=self.tools,
                            title='Accuracy',
                            x_axis_label='Epoch')
        epoch_training_plt_data = ColumnDataSource(
            data=dict(x=list(solver.epoch_training_accuracy_history.keys()),
                      y=list(solver.epoch_training_accuracy_history.values())))

        epoch_training_plt_settings = {
            'source': epoch_training_plt_data,
            'line_color': 'orange',
            'line_width': 2,
            'legend_label': 'Training'
        }
        if BOKEH_VERSION != '1.4.0':
            epoch_training_plt_settings['legend'] = epoch_training_plt_settings['legend_label']
            del epoch_training_plt_settings['legend_label']

        epoch_plot.line('x', 'y', **epoch_training_plt_settings)
        epoch_training_plt_settings['fill_color'] = epoch_training_plt_settings['line_color']
        epoch_plot.circle('x', 'y', **epoch_training_plt_settings)

        epoch_validation_plt_data = ColumnDataSource(
            data=dict(x=list(solver.epoch_validation_accuracy_history.keys()),
                      y=list(solver.epoch_validation_accuracy_history.values())))

        epoch_validation_plt_settings = {
            'source': epoch_validation_plt_data,
            'line_color': 'green',
            'line_width': 2,
            'legend_label': 'Validation'
        }
        if BOKEH_VERSION != '1.4.0':
            epoch_validation_plt_settings['legend'] = epoch_validation_plt_settings['legend_label']
            del epoch_validation_plt_settings['legend_label']

        epoch_plot.line('x', 'y', **epoch_validation_plt_settings)
        epoch_validation_plt_settings['fill_color'] = epoch_validation_plt_settings['line_color']
        epoch_plot.circle('x', 'y', **epoch_validation_plt_settings)

        epoch_plot.legend.click_policy = 'hide'
        epoch_plot.legend.location = 'bottom_right'

        epoch_handle = show(epoch_plot, notebook_handle=True)

        self.training_handle = training_handle
        self.loss_plt_data = loss_plt_data
        self.accuracy_plt_data = accuracy_plt_data
        self.epoch_handle = epoch_handle
        self.epoch_training_plt_data = epoch_training_plt_data
        self.epoch_validation_plt_data = epoch_validation_plt_data

    def append_training_loss(self, iteration, value):
        """
        Appends a value to the training loss history plot

        :param iteration:
        :param value:
        :return:
        """
        self.loss_plt_data.stream(dict(y=[value], x=[iteration]))

        push_notebook(handle=self.training_handle)

    def append_training_accuracy(self, iteration, value):
        """
        Appends a value to the training accuracy history plot

        :param iteration:
        :param value:
        :return:
        """
        self.accuracy_plt_data.stream(dict(y=[value], x=[iteration]))

        push_notebook(handle=self.training_handle)

    def append_epoch_training_accuracy(self, epoch, value):
        """
        Appends a value to the per epoch training accuracy history plot

        :param epoch:
        :param value:
        :return:
        """
        self.epoch_training_plt_data.stream(dict(y=[value], x=[epoch]))
        push_notebook(handle=self.epoch_handle)

    def append_epoch_validation_accuracy(self, epoch, value):
        """
        Appends a value to the per epoch validation accuracy history plot

        :param epoch:
        :param value:
        :return:
        """
        self.epoch_validation_plt_data.stream(dict(y=[value], x=[epoch]))
        push_notebook(handle=self.epoch_handle)


def print_bokeh_plots(solver):
    """
    Prints the plots

    :param solver:
    :return:
    """
    output_notebook()

    self = SolverPlotter(solver)
