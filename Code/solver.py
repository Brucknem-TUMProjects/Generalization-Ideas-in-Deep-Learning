"""
Nice solver my Marcel Bruckner.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from bokeh.io import output_notebook, push_notebook, show
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, LinearAxis, Range1d
from bokeh.plotting import figure
from IPython import display

import data_visualization
import optimizer


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
                 num_epochs=10,
                 verbose=True,
                 log_every=100,
                 plot=False):
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

        self.optimizer = optim
        self.criterion = criterion
        self.optim_config = optim_config
        self.lr_decay = lr_decay
        self.num_epochs = num_epochs

        self.verbose = verbose
        self.log_every = log_every
        self.plot = plot

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

        if self.validationloader:
            self.best_val_acc = 0
            self.best_params = self.model.parameters()
            self.val_acc_history = [0]

        self.loss_history = [np.nan]
        self.per_iteration_train_acc_history = [0]
        self.per_epoch_train_acc_history = [0]

        # Make a deep copy of the optim_config for each parameter
        # self.optim_configs = {}
        # for p in self.model.params:
        # d = {k: v for k, v in self.optim_config.items()}
        # self.optim_configs[p] = d

        if self.verbose:
            self.output_buffer = ""

        if self.plot:
            self.line_width = 2
            output_notebook()
            self.tools = "pan,wheel_zoom,box_zoom,reset,save,crosshair"

    def init_training_graphs(self):
        loss_plt = figure(plot_width=450,
                          plot_height=400,
                          tools=self.tools,
                          title='Training Loss',
                          x_axis_label='Iteration')
        loss_plt_data = ColumnDataSource(data=dict(x=[], y=[]))
        loss_plt.line('x',
                      'y',
                      source=loss_plt_data,
                      line_color='red',
                      line_width=self.line_width)
        # loss_handle = show(loss_plt, notebook_handle=True)

        accuracy_plt = figure(plot_width=450,
                              plot_height=400,
                              tools=self.tools,
                              title='Training Accuracy',
                              x_axis_label='Iteration')
        accuracy_plt_data = ColumnDataSource(data=dict(x=[], y=[]))
        accuracy_plt.line('x',
                          'y',
                          source=accuracy_plt_data,
                          line_color='blue',
                          line_width=self.line_width)
        # accuracy_handle = show(accuracy_plt, notebook_handle=True)

        training_handle = show(row(loss_plt, accuracy_plt),
                               notebook_handle=True)

        if self.validationloader:
            epoch_plot = figure(plot_width=900,
                                plot_height=400,
                                tools=self.tools,
                                title='Accuracy',
                                x_axis_label='Epoch')
            epoch_plot_data = ColumnDataSource(data=dict(x=[], y=[]))
            epoch_plot.line('x',
                            'y',
                            source=epoch_plot_data,
                            line_color='red',
                            line_width=self.line_width)
            epoch_handle = show(epoch_plot, notebook_handle=True)

        return training_handle, loss_plt_data, accuracy_plt_data, epoch_handle, epoch_plot_data

    def train(self):
        """ Trains the network.
        Iterates over the trainings data num_epochs time and performs gradient descent
        based on the optimizer.
        """

        device = self.device
        header = "[epoch, iteration] training loss | training accuracy"

        optim_config = self.optim_config

        if self.plot:
            training_handle, training_loss_plt_data, training_accuracy_plt_data, epoch_handle, epoch_plot_data = self.init_training_graphs(
            )

        for epoch in range(
                self.num_epochs):  # loop over the dataset multiple times

            if self.verbose:
                self.print_and_buffer(header)
                self.print_and_buffer(len(header) * "-")

            optimizer, next_lr = self.optimizer(self.model.parameters(),
                                                optim_config)
            optim_config['lr'] = next_lr
            criterion = self.criterion()

            running_loss = 0.0
            running_training_accuracy = 0.0
            running_epoch_training_accuracy = 0

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

                self.loss_history.append(np.nan)
                self.per_iteration_train_acc_history.append(np.nan)

                # print statistics
                running_loss += loss.item()
                running_training_accuracy += train_acc
                running_epoch_training_accuracy += train_acc

                if i % self.log_every == (self.log_every - 1):
                    avg_loss = running_loss / self.log_every
                    avg_acc = running_training_accuracy / self.log_every

                    del self.loss_history[-1]
                    self.loss_history.append(avg_loss)
                    del self.per_iteration_train_acc_history[-1]
                    self.per_iteration_train_acc_history.append(avg_acc)

                    training_loss_plt_data.stream(
                        dict(y=[running_loss],
                             x=[epoch * len(self.trainloader) + i]))
                    training_accuracy_plt_data.stream(
                        dict(y=[running_training_accuracy],
                             x=[epoch * len(self.trainloader) + i]))
                    push_notebook(handle=training_handle)

                    running_loss = 0.0
                    running_training_accuracy = 0.0

                    if self.verbose:
                        self.print_and_buffer(
                            '[%5d, %9d] %13.8f | %17.8f' %
                            (epoch + 1, i + 1, avg_loss, avg_acc))

            self.per_epoch_train_acc_history.append(
                running_epoch_training_accuracy / len(self.trainloader))

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
                self.val_acc_history.append(val_accuracy)

                if val_accuracy > self.best_val_acc:
                    self.best_val_acc = val_accuracy
                    self.best_params = self.model.state_dict()

                if self.verbose:
                    self.print_and_buffer(len(header) * "-")
                    self.print_and_buffer(
                        '[%5d, %9s] %13s | %17.8f' %
                        (epoch + 1, "finished", "accuracy:", val_accuracy))

            if self.verbose:
                self.print_and_buffer()


#            if self.plot:
#               self.update_plot()

    def update_plot(self):
        display.clear_output(wait=True)
        self.print_plots()
        print(self.output_buffer)

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

        if not self.validationloader:
            print(
                "Couldn't calculate accuracies as no validation set was given."
            )

            return

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
            print('Accuracy of %5s : %2d %%' %
                  (classes[i], 100 * class_correct[i] / class_total[i]))

    # Visualization

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

    def print_plots(self):
        """ Prints the plots for
            - Training loss
            - Per iteration Training accuracy
            - Validation accuracy
            - Per epoch Training accuracy
        """

        plt.subplot(2, 1, 1)
        plt.title('Training')

        # loss_history, acc_history = self.scale_iterations()

        plt.plot(self.loss_history, 'o', label='loss')
        plt.plot(self.per_iteration_train_acc_history, 'o', label='accuracy')
        plt.xlabel('Iteration')
        plt.legend(loc='lower right')

        plt.subplot(2, 1, 2)
        plt.title('Accuracy')
        plt.plot(self.per_epoch_train_acc_history, '-o', label='train')

        if self.validationloader:
            plt.plot(self.val_acc_history, '-o', label='val')
        # plt.plot([0.5] * len(self.val_acc_history), 'k--')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.gcf().set_size_inches(15, 12)
        plt.show()

    def data_shader_print_plots(self):
        output_notebook()

        p = figure(plot_height=400,
                   plot_width=900,
                   title="Training",
                   x_axis_label="Iteration")

        p.line(x=range(len(self.loss_history)),
               y=self.loss_history,
               legend="Loss",
               line_color="red",
               line_width=2)
        p.y_range = Range1d(0, np.nanmax(self.loss_history))

        p.extra_y_ranges = {
            'Accuracy':
            Range1d(0, np.nanmax(self.per_iteration_train_acc_history))
        }
        p.add_layout(LinearAxis(y_range_name='Accuracy'), 'right')
        p.line(x=range(len(self.per_iteration_train_acc_history)),
               y=self.per_iteration_train_acc_history,
               legend="Accuracy",
               line_color="blue",
               line_width=2,
               y_range_name='Accuracy')

        show(p)

        p = figure(plot_height=400,
                   plot_width=900,
                   title="Training",
                   x_axis_label="Accuracy")

        p.circle(x=range(len(self.per_epoch_train_acc_history)),
                 y=self.per_epoch_train_acc_history,
                 legend="Training",
                 line_color="red",
                 line_width=2)
        p.line(x=range(len(self.per_epoch_train_acc_history)),
               y=self.per_epoch_train_acc_history,
               legend="Training",
               line_color="red",
               line_width=2)
        p.y_range = Range1d(0, np.nanmax(self.per_epoch_train_acc_history))

        p.extra_y_ranges = {
            'Validation': Range1d(0, np.nanmax(self.val_acc_history))
        }
        p.add_layout(LinearAxis(y_range_name='Validation'), 'right')
        p.circle(x=range(len(self.val_acc_history)),
                 y=self.val_acc_history,
                 legend="Validation",
                 line_color="blue",
                 line_width=2,
                 y_range_name='Validation')
        p.line(x=range(len(self.val_acc_history)),
               y=self.val_acc_history,
               legend="Validation",
               line_color="blue",
               line_width=2,
               y_range_name='Validation')

        show(p)

    def print_and_buffer(self, message=""):
        if self.verbose:
            self.output_buffer += message + "\n"
        print(message)
