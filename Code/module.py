"""
CLI trainer
"""
# include standard modules
import argparse
import os
import shutil

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
PARSER.add_argument("--empty_dir",
                    "-ed",
                    help="set to delete all files in the specified directory (default: False)",
                    action="store_true")

FILENAME = 'solver.pth'
PARSER.add_argument("--file",
                    "-f",
                    help="specify the filename (default: %s)" % FILENAME)

KNOWN_SETS = ['cifar10']
TRAIN_SET = KNOWN_SETS[0]
PARSER.add_argument("--train_set",
                    "-ts",
                    help="specify the training set (default: %s)" % TRAIN_SET)
VALIDATION_SET = KNOWN_SETS[0]
PARSER.add_argument("--validation_set",
                    "-vs",
                    help="specify the validation set (default: %s)" %
                    VALIDATION_SET)

BATCH_SIZE = 64
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
                    "-e",
                    help="specify the number of epochs (default: %s)" %
                    NUM_EPOCHS)
SAVE_EVERY = -1
PARSER.add_argument(
    "--save_every_nth_epoch",
    "-nth",
    help=
    "specify after how many epochs the solver save its state to drive (default: %s)"
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
PARSER.add_argument("--save_latest",
                    "-sl",
                    help="save the solver after every epoch",
                    action="store_true")
PARSER.add_argument("--debug", help="debug script", action="store_true")

# read arguments from the command line
ARGS = PARSER.parse_args()


def train():
    if ARGS.empty_dir:
        shutil.rmtree(FOLDER, ignore_errors=True)
        ARGS.create_solver = True

    if ARGS.create_solver:
        solver = Solver()
    else:
        solver = load_solver(folder=FOLDER,
                             filename=FILENAME)

    train_loader = getattr(data_loader, TRAIN_SET)(True, BATCH_SIZE,
                                                   SUBSET_SIZE, RANDOM_LABELS)

    if VALIDATE:
        validation_loader = getattr(data_loader,
                                    TRAIN_SET)(False, BATCH_SIZE, SUBSET_SIZE,
                                               RANDOM_LABELS)
    else:
        validation_loader = None

    solver.train(model=MODEL,
                 trainings_loader=train_loader,
                 validation_loader=validation_loader,
                 training={
                     'epochs': NUM_EPOCHS,
                     'log_every': LOG_EVERY,
                     'plot': PLOT,
                     'verbose': VERBOSE
                 },
                 saving={
                     'latest': True,
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
SUBSET_SIZE = int(
    ARGS.subset_size if ARGS.subset_size is not None else SUBSET_SIZE)

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
MODEL = getattr(networks, MODEL)()

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

if ARGS.train:
    train()

print("Module finished!")
print("\n" + 80 * "*" + "\n")
