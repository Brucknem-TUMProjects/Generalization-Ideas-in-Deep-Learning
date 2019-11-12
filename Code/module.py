"""
CLI trainer
"""
# include standard modules
import argparse
import os

import data_loader
import data_visualization
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
FILENAME = 'solver.pth'
PARSER.add_argument("--file",
                    "-f",
                    help="specify the filename (default: %s)" % FILENAME)

KNOWN_SETS = ['cifar10']
TRAIN_SET = KNOWN_SETS[0]
PARSER.add_argument("--train_set",
                    help="specify the training set (default: %s)" % TRAIN_SET)
VALIDATION_SET = KNOWN_SETS[0]
PARSER.add_argument("--validation_set",
                    help="specify the validation set (default: %s)" %
                    VALIDATION_SET)

BATCH_SIZE = 64
PARSER.add_argument("--batch_size",
                    help="specify the batch size (default: %s)" % BATCH_SIZE)
SUBSET_SIZE = -1
PARSER.add_argument(
    "--subset_size",
    help="specify the subset size, -1 for whole set (default: %s)" %
    SUBSET_SIZE)
PARSER.add_argument("--random_labels",
                    "-r",
                    help="use randomly generated labels (default: %s)" % False,
                    action="store_true")

MODEL = 'ExampleNet'
PARSER.add_argument("--model",
                    "-m",
                    help="specify the model (default: ExampleNet)")
PARSER.add_argument("--train",
                    "-t",
                    help="train the model",
                    action="store_true")
PARSER.add_argument("--validate",
                    "-v",
                    help="validate accuracy after each epoch",
                    action="store_true")

NUM_EPOCHS = 5
PARSER.add_argument("--num_epochs",
                    help="specify the number of epochs (default: %s)" %
                    NUM_EPOCHS)
SAVE_EVERY = -1
PARSER.add_argument(
    "--save_every_epoch",
    help=
    "specify after how many epochs the solver save its state to drive (default: %s)"
    % SAVE_EVERY)
LOG_EVERY = 50
PARSER.add_argument(
    "--log_every_iteration",
    help=
    "specify after how many iterations the solver logs the current loss and accuracy (default: %s)"
    % LOG_EVERY)
PARSER.add_argument("--verbose",
                    "-vv",
                    help="specify if the progress is printed to screen",
                    action="store_true")
PARSER.add_argument("--debug", help="debug script", action="store_true")

# read arguments from the command line
ARGS = PARSER.parse_args()


def create_solver():
    model = getattr(networks, MODEL)()
    print("Creating a new solver %s%s with model: %s" %
          (FOLDER, FILENAME, model))
    print("\n" + 80 * "*" + "\n")
    Solver(model).save_solver(filename=FILENAME, folder=FOLDER)


def train():
    print(
        "Loading training set: %s, Batch size: %s, Subset size: %s, Random labels: %s"
        % (TRAIN_SET, BATCH_SIZE, SUBSET_SIZE, RANDOM_LABELS))
    train_loader = getattr(data_loader, TRAIN_SET)(True, BATCH_SIZE,
                                                   SUBSET_SIZE, RANDOM_LABELS)
    print("\n" + 80 * "*" + "\n")

    if VALIDATE:
        print(
            "Loading validation set: %s, Batch size: %s, Subset size: %s, Random labels: %s"
            % (TRAIN_SET, BATCH_SIZE, SUBSET_SIZE, RANDOM_LABELS))
        validation_loader = getattr(data_loader,
                                    TRAIN_SET)(False, BATCH_SIZE, SUBSET_SIZE,
                                               RANDOM_LABELS)
        print("\n" + 80 * "*" + "\n")
    else:
        validation_loader = None

    print("Loading solver: %s%s" % (FOLDER, FILENAME))
    solver = load_solver(trainloader=train_loader,
                         validationloader=validation_loader,
                         folder=FOLDER,
                         filename=FILENAME)
    print("\n" + 80 * "*" + "\n")

    print("Training starts! \nEpochs: %s, Logging every %s iterations\n" %
          (NUM_EPOCHS, LOG_EVERY))
    solver.train(num_epochs=NUM_EPOCHS,
                 log_every=LOG_EVERY,
                 plot=PLOT,
                 verbose=VERBOSE,
                 save_after_epoch=True,
                 save_every_epoch=SAVE_EVERY,
                 save_best_solver=True,
                 folder=FOLDER,
                 filename=FILENAME)
    print(80 * "*" + "\n")
    print("Saving solver!")
    solver.save_solver(filename=FILENAME, folder=FOLDER)
    print("\n" + 80 * "*" + "\n")


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
    ARGS.save_every_epoch if ARGS.save_every_epoch is not None else SAVE_EVERY)
SUBSET_SIZE = int(
    ARGS.subset_size if ARGS.subset_size is not None else SUBSET_SIZE)

FOLDER = ARGS.dir if ARGS.dir is not None else FOLDER
FILENAME = ARGS.file if ARGS.file is not None else FILENAME

print(SAVE_EVERY)

if not FOLDER.endswith('/'):
    FOLDER += '/'

if not FILENAME.endswith('.pth'):
    FILENAME += '.pth'

if ARGS.model:
    if not hasattr(networks, ARGS.model):
        raise ValueError('Invalid model: "%s"' % ARGS.model)
    MODEL = ARGS.model

if ARGS.train_set:
    if not hasattr(data_loader, ARGS.train_set):
        raise ValueError('Invalid trainset: "%s"' % ARGS.train_set)
    TRAIN_SET = ARGS.train_set

if ARGS.validation_set:
    if not hasattr(data_loader, ARGS.validation_set):
        raise ValueError('Invalid validation set: "%s"' % ARGS.validation_set)
    TRAIN_SET = ARGS.validation_set

print("\n" + 80 * "*" + "\n")
print("Module is running!")
print("\n" + 80 * "*" + "\n")

if ARGS.create_solver:
    create_solver()

if ARGS.train:
    train()

print("Module finished!")
print("\n" + 80 * "*" + "\n")
