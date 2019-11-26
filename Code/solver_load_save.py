import _pickle as cPickle
import os

import helpers
import solver as _solver

APPENDIX = 'a'


def save_best_solver(solver, filename='solver_best.pth', folder='solvers', verbose=False):
    """
    Saves the best solver

    :param solver:
    :param filename:
    :param folder:
    :param verbose:
    :return:
    """
    if solver.best_solver:
        filename = remove_pth(filename) + '_best.pth'
        save_solver(solver.best_solver, filename, folder, verbose)
    else:
        print('No best solver present. Maybe missing a validation loader!')


def save_solver(solver, filename='solver.pth', folder='solvers', epoch='', verbose=False):
    """
    Saves the solver

    :param solver:
    :param filename:
    :param folder:
    :param epoch:
    :param verbose:
    :return:
    """
    solver.model.train()
    folder = folder if folder.endswith('/') else folder + '/'

    if not os.path.exists(folder):
        os.makedirs(folder)

    if epoch:
        filename = remove_pth(filename)
        filename += '_e' + str(epoch)

    filename = add_pth(filename)

    if verbose:
        helpers.print_separated("Saving solver to %s%s" % (folder, filename))

    # print(solver_visualization.debug_sizes(solver, epoch))
    cPickle.dump(dict(solver.__dict__), open(folder + filename, 'wb'), 2)


def load_solver(filename='solver.pth',
                folder='solvers',
                load_latest_if_not_found=True,
                verbose=True):
    """
    Loads a solver

    :param filename:
    :param folder:
    :param verbose:
    :return:
    """
    folder = folder if folder.endswith('/') else folder + '/'

    if verbose:
        helpers.print_separated("Loading solver from %s%s" % (folder, filename))

    if not os.path.exists(folder + filename) and load_latest_if_not_found:
        from os import walk

        f = []
        for (dirpath, dirnames, filenames) in walk(folder):
            f.extend(filenames)
            break
        filename = min(f, key=len)
    else:
        raise ValueError('No solver %s%s found.' % folder, filename)

    data = cPickle.load(open(folder + filename, 'rb'))

    return _solver.Solver(**data)


def add_pth(s):
    """
    Adds '.pth' to the string if not already present

    :param s:
    :return:
    """
    return s if s.endswith('.pth') else s + '.pth'


def remove_pth(s):
    """
    Removes '.pth' from the string if present

    :param s:
    :return:
    """
    return s if not s.endswith('.pth') else s[:-len('.pth')]
