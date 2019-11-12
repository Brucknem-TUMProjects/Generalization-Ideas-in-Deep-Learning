import bokeh
import matplotlib.pyplot as plt
import numpy as np
import torch
from bokeh.io import output_notebook, push_notebook, show
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, LinearAxis, Range1d
from bokeh.plotting import figure
from IPython import display

BOKEH_VERSION = bokeh.__version__


class SolverPrinter:
    def __init__(self, verbose=False):
        self.verbose = verbose


def print_class_accuracies(solver, classes=None):
    """ Prints the class accuracies.
    classes -- list of classnames - (default=None)
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


class SolverPlotter:
    def __init__(self, solver):
        output_notebook()
        self.tools = "pan,wheel_zoom,box_zoom,reset,save,crosshair, hover"

        loss_plt = figure(plot_width=450,
                          plot_height=400,
                          tools=self.tools,
                          title='Training Loss',
                          x_axis_label='Iteration')
        loss_plt_data = ColumnDataSource(
            data=dict(x=list(solver.loss_history.keys()),
                      y=list(solver.loss_history.values())))
        loss_plt.line('x',
                      'y',
                      source=loss_plt_data,
                      line_color='red',
                      line_width=1)

        accuracy_plt = figure(plot_width=450,
                              plot_height=400,
                              tools=self.tools,
                              title='Training Accuracy',
                              x_axis_label='Iteration')
        accuracy_plt_data = ColumnDataSource(
            data=dict(x=list(solver.per_iteration_train_acc_history.keys()),
                      y=list(solver.per_iteration_train_acc_history.values())))
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
            data=dict(x=list(solver.per_epoch_train_acc_history.keys()),
                      y=list(solver.per_epoch_train_acc_history.values())))

        if BOKEH_VERSION == '1.4.0':
            epoch_plot.line('x',
                            'y',
                            source=epoch_training_plt_data,
                            line_color='orange',
                            line_width=2,
                            legend_label='Training')
            epoch_plot.circle('x',
                              'y',
                              source=epoch_training_plt_data,
                              line_color='orange',
                              fill_color='orange',
                              line_width=2,
                              legend_label='Training')
        else:
            epoch_plot.line('x',
                            'y',
                            source=epoch_training_plt_data,
                            line_color='orange',
                            line_width=2,
                            legend='Training')
            epoch_plot.circle('x',
                              'y',
                              source=epoch_training_plt_data,
                              line_color='orange',
                              fill_color='orange',
                              line_width=2,
                              legend='Training')

        epoch_validation_plt_data = ColumnDataSource(
            data=dict(x=list(solver.val_acc_history.keys()),
                      y=list(solver.val_acc_history.values())))
        if BOKEH_VERSION == '1.4.0':
            epoch_plot.circle('x',
                              'y',
                              source=epoch_validation_plt_data,
                              line_color='green',
                              fill_color='green',
                              line_width=2,
                              legend_label='Validation')
            epoch_plot.line('x',
                            'y',
                            source=epoch_validation_plt_data,
                            line_color='green',
                            line_width=2,
                            legend_label='Validation')
        else:
            epoch_plot.circle('x',
                              'y',
                              source=epoch_validation_plt_data,
                              line_color='green',
                              fill_color='green',
                              line_width=2,
                            legend='Validation')
            epoch_plot.line('x',
                            'y',
                            source=epoch_validation_plt_data,
                            line_color='green',
                            line_width=2,
                            legend='Validation')
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
        self.loss_plt_data.stream(dict(y=[value], x=[iteration]))

        push_notebook(handle=self.training_handle)

    def append_training_accuracy(self, iteration, value):
        self.accuracy_plt_data.stream(dict(y=[value], x=[iteration]))

        push_notebook(handle=self.training_handle)

    def append_epoch_training_accuracy(self, epoch, value):
        self.epoch_training_plt_data.stream(dict(y=[value], x=[epoch]))
        push_notebook(handle=self.epoch_handle)

    def append_epoch_validation_accuracy(self, epoch, value):
        self.epoch_validation_plt_data.stream(dict(y=[value], x=[epoch]))
        push_notebook(handle=self.epoch_handle)

    def print_plots(self, solver):
        """ Prints the plots for
            - Training loss
            - Per iteration Training accuracy
            - Validation accuracy
            - Per epoch Training accuracy
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
        # plt.plot([0.5] * len(solver.val_acc_history), 'k--')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.gcf().set_size_inches(15, 12)
        plt.show()


def print_bokeh_plots(solver):
    output_notebook()

    self = SolverPlotter(solver)
