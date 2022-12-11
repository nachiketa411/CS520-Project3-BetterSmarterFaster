import random

import numpy as np

from Constants import ALPHA, NO_OF_HIDDEN_UNITS, NO_OF_HIDDEN_LAYERS

# NEED TO CHECK THE INDICES ISSUE

class NeuralNetwork:
    def __init__(self, input_size, activation_function, gradient_activation,
                 loss_function, gradient_loss, batch_size, no_of_epochs):

        np.random.seed(1)
        # hyperparameter initialization
        self.alpha = ALPHA
        self.no_of_hidden_layers = NO_OF_HIDDEN_LAYERS
        self.no_of_units = NO_OF_HIDDEN_UNITS
        self.activation_function = activation_function
        self.gradient_activation = gradient_activation
        self.loss_function = loss_function
        self.gradient_loss = gradient_loss
        self.weights = []
        self.dW = []
        self.z = []
        self.dZ = []
        self.a = []
        self.dA = []
        self.input_size = input_size
        self.batch_size = batch_size
        self.epochs = no_of_epochs
        self.loss_for_epochs = []

        # self.dataset = dataset
        # self.training_data = None
        # self.testing_data = None

        # Input vector Dim: no_of_inputs x batch_size
        self.bias = []
        self.dB = []
        for layer in range(self.no_of_hidden_layers + 1):
            self.bias.append(np.random.rand((self.no_of_units[layer], 1)))
            self.dB.append(np.zeros((self.no_of_units[layer], 1)))
            if layer == 0:
                self.weights.append(np.random.rand((self.no_of_units[layer], self.input_size)))
                self.dW.append(np.zeros((self.no_of_units[layer], self.input_size)))
                self.z.append(np.zeros((self.no_of_units[layer], self.input_size)))
                self.dZ.append(np.zeros((self.no_of_units[layer], self.input_size)))
                self.a.append(np.zeros((self.no_of_units[layer], self.input_size)))
                self.dA.append(np.zeros((self.no_of_units[layer], self.input_size)))
            else:
                self.weights.append(np.random.rand((self.no_of_units[layer], self.no_of_units[layer - 1])))
                self.dW.append(np.zeros((self.no_of_units[layer], self.no_of_units[layer - 1])))
                self.z.append(np.zeros((self.no_of_units[layer], self.no_of_units[layer - 1])))
                self.dZ.append(np.zeros((self.no_of_units[layer], self.no_of_units[layer - 1])))
                self.a.append(np.zeros((self.no_of_units[layer], self.no_of_units[layer - 1])))
                self.dA.append(np.zeros((self.no_of_units[layer], self.no_of_units[layer - 1])))

    # Input Vector Dims: input_size x Batch_Size
    # Expected Output Dims: Batch_Size x 1
    # X = [X(1) X(2) X(3)...X(Batch_size)]
    # Z[1] = W[1] @ X
    # A[1] = Activation_function(Z[1])
    #      = [A(1) A(2) A(3)...A(Batch_size)]
    # The first column represents the activation value for the first input, similarly the 2nd column represents
    # the activation for the 2nd input and so on
    def forward_propagation(self, input_vector):
        self.a[0] = input_vector
        for layer in range(self.no_of_hidden_layers + 1):
            self.z[layer + 1] = (self.weights[layer] @ self.a[layer]) + self.bias[layer]
            self.a[layer + 1] = self.activation_function(self.z[layer])

    # dL = dL/dA[Last layer]
    def backward_propagate(self, expected_output):
        for layer in range(self.no_of_hidden_layers, -1, -1):
            if layer == self.no_of_hidden_layers:
                self.dA[layer] = self.gradient_loss(self.a[layer], expected_output)
            else:
                self.dA[layer] = self.dZ[layer + 1] @ self.weights[layer + 1]
            self.dZ[layer] = (self.dA[layer] * self.gradient_activation(self.z[layer]))
            self.dW[layer] = np.dot(self.dZ[layer], self.a[layer - 1]) / self.batch_size
            self.dB[layer] = np.sum(self.dZ[layer], axis=1, keepdims=True) / self.batch_size

            self.weights[layer] = self.weights[layer] - self.alpha * self.dW[layer]
            self.bias[layer] = self.bias[layer] - self.alpha * self.dB[layer]

            # dZ[self.no_of_hidden_layers - i] = dL

    def fit(self, input_vector, expected_output):
        for epoch in range(self.epochs):
            self.forward_propagation(input_vector)
            self.backward_propagate(expected_output)

            self.loss_for_epochs.append(self.loss_function(self.a[self.no_of_hidden_layers], expected_output))
            print('Loss for Epoch: ', epoch, ' is: ', self.loss_for_epochs[epoch])











