import os
import solver as solv
import _pickle as cPickle
import helpers

import solver_visualization

APPENDIX = 'a'

def save_best_solver(solver, filename='solver_best.pth', folder='solvers', verbose=False):
    if solver.best_solver:
        filename = remove_pth(filename) + '_best.pth'
        save_solver(solver.best_solver, filename, folder, verbose)
    else:
        print('No best solver present. Maybe missing a validation loader!')


def save_solver(solver, filename='solver.pth', folder='solvers', epoch='', verbose=False):
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
                verbose=True):
    folder = folder if folder.endswith('/') else folder + '/'

    if verbose:
        helpers.print_separated("Loading solver from %s%s" % (folder, filename))

    data = cPickle.load(open(folder + filename, 'rb'))

    return solv.Solver(**data)


def add_pth(s):
    return s if s.endswith('.pth') else s + '.pth'


def remove_pth(s):
    return s if not s.endswith('.pth') else s[:-len('.pth')]
