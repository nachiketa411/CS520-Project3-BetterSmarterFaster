# np.vectorize will let us use the scalar function over a vector of inputs
import math

import numpy as np


def linear(z):
    return z


def gradient_linear(z):
    return np.ones(z.shape)
# ----------------------------------------------------------------------------------


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))
# ----------------------------------------------------------------------------------


def tan_h(z):
    exp_z = np.exp(z)
    exp_neg_z = np.exp(-z)
    return (exp_z - exp_neg_z) / (exp_z + exp_neg_z)


def gradient_tan_h(z):
    return 1 - tan_h(z) ** 2
# ----------------------------------------------------------------------------------


def rel_u(z):
    return np.maximum(0, z)


def gradient_rel_u(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z
# ----------------------------------------------------------------------------------


def leaky_rel_u(z):
    return np.maximum(0.01 * z, z)


def gradient_leaky_rel_u(z):
    z[z < 0] = 0.01
    z[z >= 0] = 1
    return z


