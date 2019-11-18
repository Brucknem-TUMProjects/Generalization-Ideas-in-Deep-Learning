"""
CLI trainer
"""
# include standard modules
import argparse
import shutil

import networks
from solver import *

# initiate the parser
PARSER = argparse.ArgumentParser()

# add long and short argument
PARSER.add_argument("--create_solver",
                    "-c",
                    help="create a new solver",
                    action="store_true")

FOLDER = 'solvers/'
PARSER.add_argument("--dir",
                    "-d",
                    help="specify the directory (default: %s)" % FOLDER)
PARSER.add_argument("--empty_dir",
                    "-ed",
                    help="set to delete all files in the specified directory (default: False)",
                    action="store_true")

FILENAME = 'solver.pth'
PARSER.add_argument("--file",
                    "-f",
                    help="specify the filename (default: %s)" % FILENAME)

KNOWN_SETS = ['cifar10']
DATA_SET = KNOWN_SETS[0]
PARSER.add_argument("--data_set",
                    "-ds",
                    help="specify the data set (default: %s)" % DATA_SET)

BATCH_SIZE = 4
PARSER.add_argument("--batch_size",
                    "-bs",
                    help="specify the batch size (default: %s)" % BATCH_SIZE)
SUBSET_SIZE = -1
PARSER.add_argument(
    "--subset_size",
    "-ss",
    help="specify the subset size, -1 for whole set (default: %s)" %
         SUBSET_SIZE)
PARSER.add_argument("--random_labels",
                    "-r",
                    help="use randomly generated labels",
                    action="store_true")

MODEL = 'vgg16_bn'
PARSER.add_argument("--model",
                    "-m",
                    help="specify the model (default: %s)" % MODEL)
PARSER.add_argument("--train",
                    "-t",
                    help="train the model",
                    action="store_true")
PARSER.add_argument("--validate",
                    "-v",
                    help="validate accuracy after each epoch",
                    action="store_true")

HIDDEN_UNITS = 32
PARSER.add_argument("--hidden_units",
                    "-hu",
                    help="specify the number of hidden units for the TwoLayerPerceptron model (default: %s)" %
                         HIDDEN_UNITS)
NUM_EPOCHS = -1
PARSER.add_argument("--num_epochs",
                    "-e",
                    help="specify the number of epochs (default: %s)" %
                         NUM_EPOCHS)
SAVE_EVERY = -1
PARSER.add_argument(
    "--save_every_nth_epoch",
    "-nth",
    help=
    "specify after how many epochs the solver saves a checkpoint of its state to drive (default: %s)"
    % SAVE_EVERY)
LOG_EVERY = 50
PARSER.add_argument(
    "--log_every_iteration",
    "-l",
    help=
    "specify after how many iterations the solver logs the current loss and accuracy (default: %s)"
    % LOG_EVERY)
PARSER.add_argument("--verbose",
                    "-vv",
                    help="specify if the progress is printed to screen",
                    action="store_true")
PARSER.add_argument("--save_best",
                    "-sb",
                    help="save the solver whenever it exceeds the best validation accuracy",
                    action="store_true")
PARSER.add_argument("--save_on_training_100",
                    "-st100",
                    help="save the solver when it hits 100% training accuracy",
                    action="store_true")
SAVE_LATEST = -1
PARSER.add_argument("--save_latest",
                    "-sl",
                    help="save the solver state after every nth epoch (default: %s)" % SAVE_LATEST)
CONFUSION_SET_SIZE = 0
PARSER.add_argument("--confusion_set_size",
                    "-cs",
                    help="number of additional, randomly labelled data points to confuse the model (default: %s)" %
                         CONFUSION_SET_SIZE)
PARSER.add_argument("--debug", help="debug script", action="store_true")

# read arguments from the command line
ARGS = PARSER.parse_args()


def train():
    """
    Trains a network

    :return:
    """
    if ARGS.empty_dir:
        shutil.rmtree(FOLDER, ignore_errors=True)
        ARGS.create_solver = True

    if ARGS.create_solver:
        solver = Solver(
            model=MODEL,
            strategy={
                'optimizer': 'adam',
                'criterion': 'cross_entropy_loss',
                'optimizer_config': {},
                'lr_decay': 0.9
            },
            data={
                'dataset': DATA_SET,
                'batch_size': BATCH_SIZE,
                'subset_size': SUBSET_SIZE,
                'random_labels': RANDOM_LABELS,
                'confusion_set_size': CONFUSION_SET_SIZE
            }
        )
    else:
        solver = load_solver(folder=FOLDER,
                             filename=FILENAME)

    solver.train(
        training={
            'epochs': NUM_EPOCHS,
            'log_every': LOG_EVERY,
            'plot': PLOT,
            'verbose': VERBOSE,
            'save_on_training_100': ARGS.save_on_training_100,
            'validate': ARGS.validate
        },
        saving={
            'latest': SAVE_LATEST,
            'nth_epoch': SAVE_EVERY,
            'best': ARGS.save_best,
            'folder': FOLDER,
            'filename': FILENAME
        })

    solver.save_solver(filename=FILENAME, folder=FOLDER)


VALIDATE = ARGS.validate
RANDOM_LABELS = ARGS.random_labels
PLOT = False
VERBOSE = ARGS.verbose

BATCH_SIZE = int(
    ARGS.batch_size if ARGS.batch_size is not None else BATCH_SIZE)
NUM_EPOCHS = int(
    ARGS.num_epochs if ARGS.num_epochs is not None else NUM_EPOCHS)
LOG_EVERY = int(ARGS.log_every_iteration
                if ARGS.log_every_iteration is not None else LOG_EVERY)
SAVE_EVERY = int(
    ARGS.save_every_nth_epoch if ARGS.save_every_nth_epoch is not None else SAVE_EVERY)
SAVE_LATEST = int(
    ARGS.save_latest if ARGS.save_latest is not None else SAVE_LATEST)
SUBSET_SIZE = int(
    ARGS.subset_size if ARGS.subset_size is not None else SUBSET_SIZE)
CONFUSION_SET_SIZE = int(
    ARGS.confusion_set_size if ARGS.confusion_set_size is not None else CONFUSION_SET_SIZE)
HIDDEN_UNITS = int(
    ARGS.hidden_units if ARGS.hidden_units is not None else HIDDEN_UNITS)

FOLDER = ARGS.dir if ARGS.dir is not None else FOLDER
FILENAME = ARGS.file if ARGS.file is not None else FILENAME

if not FOLDER.endswith('/'):
    FOLDER += '/'

if not FILENAME.endswith('.pth'):
    FILENAME += '.pth'

if ARGS.model:
    if not hasattr(networks, ARGS.model):
        raise ValueError('Invalid model: "%s"' % ARGS.model)
    MODEL = ARGS.model
MODEL = getattr(networks, MODEL)(**{'hidden_units': HIDDEN_UNITS})

if ARGS.data_set:
    if not hasattr(data_loader, ARGS.data_set):
        raise ValueError('Invalid trainset: "%s"' % ARGS.data_set)
    DATA_SET = ARGS.data_set

if ARGS.verbose:
    print("\n" + 80 * "*" + "\n")
    print("Module is running!")
    print("\n" + 80 * "*" + "\n")

if ARGS.train:
    train()

if ARGS.verbose:
    print("Module finished!")
    print("\n" + 80 * "*" + "\n")
