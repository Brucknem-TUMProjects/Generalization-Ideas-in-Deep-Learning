import numpy as np

from norms import *

# L2 norm

layers = {0: np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]])}

assert l2_norm(layers) == 12, "Should be 12"

layers = {0: np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])}

assert l2_norm(layers) == 12, "Should be 12"

layers = {0: np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])}

assert l2_norm(layers) == 0, "Should be 0"
