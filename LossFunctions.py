import numpy as np


# Loss = |A[L] - Expected_output|
def manhattan_loss(actual_output, expected_output):
    return np.sum(abs(actual_output - expected_output))


# dLoss/dA[L] = sign (A[L] - Expected_output)
def gradient_manhattan_loss(actual_output, expected_output):
    return np.sign(actual_output - expected_output)


# Loss = norm(A[L] - expected_output)
def euclidean_loss(actual_output, expected_output):
    return np.linalg.norm(actual_output - expected_output)


# dLoss/dA[L] = (A[L] - Expected_output) / norm (A[L] - Expected_output)
def gradient_euclidean_loss(actual_output, expected_output):
    # print(np.shape(actual_output))
    # print(np.shape(expected_output))
    diff = actual_output - expected_output
    return np.divide(diff, np.linalg.norm(diff))

