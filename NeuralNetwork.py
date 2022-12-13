import random

import numpy as np

from ActivationFunction import gradient_linear, linear
from Constants import ALPHA, NO_OF_HIDDEN_UNITS, NO_OF_HIDDEN_LAYERS, BETA_1, BETA_2, EPSILON, TWO_LAYER_WEIGHTS_PATH, \
    ONE_LAYER_WEIGHTS_PATH, BETA, REWARD


# NEED TO CHECK THE INDICES ISSUE

class NeuralNetwork:
    def __init__(self, input_size, activation_function, gradient_activation,
                 loss_function, gradient_loss, batch_size, pre_trained_weights):

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

        self.m_dW = []
        self.m_dB = []

        self.v_dW = []
        self.v_dB = []

        self.input_size = input_size
        self.batch_size = batch_size
        # self.epochs = no_of_epochs
        self.loss_for_batches = []

        # Input vector Dim: no_of_inputs x batch_size
        self.bias = []
        self.dB = []
        for layer in range(self.no_of_hidden_layers + 1):
            self.bias.append(np.random.rand(self.no_of_units[layer], 1))
            self.dB.append(np.zeros((self.no_of_units[layer], 1)))

            self.m_dB.append(np.zeros((self.no_of_units[layer], 1)))
            self.v_dB.append(np.zeros((self.no_of_units[layer], 1)))
            if layer == 0:
                self.weights.append(np.random.rand(self.no_of_units[layer], self.input_size))
                self.dW.append(np.zeros((self.no_of_units[layer], self.input_size)))
                self.z.append(np.zeros((self.no_of_units[layer], self.batch_size)))
                self.dZ.append(np.zeros((self.no_of_units[layer], self.batch_size)))
                self.a.append(np.zeros((self.no_of_units[layer], self.batch_size)))
                self.dA.append(np.zeros((self.no_of_units[layer], self.batch_size)))

                self.m_dW.append(np.zeros((self.no_of_units[layer], self.input_size)))
                self.v_dW.append(np.zeros((self.no_of_units[layer], self.input_size)))
            else:
                self.weights.append(np.random.rand(self.no_of_units[layer], self.no_of_units[layer - 1]))
                self.dW.append(np.zeros((self.no_of_units[layer], self.no_of_units[layer - 1])))
                self.z.append(np.zeros((self.weights[layer].shape[0], self.a[layer - 1].shape[1])))
                self.dZ.append(np.zeros((self.weights[layer].shape[0], self.a[layer - 1].shape[1])))
                self.a.append(np.zeros((self.weights[layer].shape[0], self.a[layer - 1].shape[1])))
                self.dA.append(np.zeros((self.weights[layer].shape[0], self.a[layer - 1].shape[1])))

                self.m_dW.append(np.zeros((self.no_of_units[layer], self.no_of_units[layer - 1])))
                self.v_dW.append(np.zeros((self.no_of_units[layer], self.no_of_units[layer - 1])))

        if pre_trained_weights is not None:
            self.weights = pre_trained_weights

    # Input Vector Dims: input_size x Batch_Size
    # Expected Output Dims: Batch_Size x 1
    # X = [X(1) X(2) X(3)...X(Batch_size)]
    # Z[1] = W[1] @ X
    # A[1] = Activation_function(Z[1])
    #      = [A(1) A(2) A(3)...A(Batch_size)]
    # The first column represents the activation value for the first input, similarly the 2nd column represents
    # the activation for the 2nd input and so on
    def forward_propagation(self, input_vector):
        for layer in range(self.no_of_hidden_layers + 1):
            if layer == 0:
                self.z[layer] = (self.weights[layer] @ input_vector) + self.bias[layer]
            else:
                self.z[layer] = (self.weights[layer] @ self.a[layer - 1]) + self.bias[layer]

            if layer == self.no_of_hidden_layers:
                self.a[layer] = linear(self.z[layer])
            else:
                self.a[layer] = self.activation_function(self.z[layer])

    # dL = dL/dA[Last layer]
    def backward_propagate(self, input_vector, expected_output, epoch):
        for layer in range(self.no_of_hidden_layers, -1, -1):
            # print('LAYER: ', layer, '----------------')
            if layer == self.no_of_hidden_layers:
                self.dA[layer] = self.gradient_loss(self.a[layer], expected_output)
                self.dZ[layer] = (self.dA[layer] * gradient_linear(self.z[layer]))
            else:
                self.dA[layer] = np.dot(self.weights[layer + 1].T, self.dZ[layer + 1])
                self.dZ[layer] = (self.dA[layer] * self.gradient_activation(self.z[layer]))
            if layer == 0:
                # self.dW[layer] = np.dot(self.dZ[layer], input_vector.T) / self.batch_size
                self.dW[layer] = np.dot(self.dZ[layer], input_vector.T)
            else:
                # self.dW[layer] = np.dot(self.dZ[layer], self.a[layer - 1].T) / self.batch_size
                self.dW[layer] = np.dot(self.dZ[layer], self.a[layer - 1].T)
            # self.dB[layer] = np.sum(self.dZ[layer], axis=1, keepdims=True) / self.batch_size
            self.dB[layer] = np.sum(self.dZ[layer], axis=1, keepdims=True)

            self.m_dW[layer] = (BETA_1 * self.m_dW[layer]) + ((1 - BETA_1) * self.dW[layer])
            self.v_dW[layer] = (BETA_2 * self.v_dW[layer]) + ((1 - BETA_2) * np.square(self.dW[layer]))

            self.m_dB[layer] = (BETA_1 * self.m_dB[layer]) + ((1 - BETA_1) * self.dB[layer])
            self.v_dB[layer] = (BETA_2 * self.v_dB[layer]) + ((1 - BETA_2) * np.square(self.dB[layer]))

            m_dw_bias_corrected = np.divide(self.m_dW[layer], 1 - (BETA_1 ** epoch))
            v_dw_bias_corrected = np.divide(self.v_dW[layer], 1 - (BETA_2 ** epoch))

            m_db_bias_corrected = np.divide(self.m_dB[layer], 1 - (BETA_1 ** epoch))
            v_db_bias_corrected = np.divide(self.v_dB[layer], 1 - (BETA_2 ** epoch))

            # self.weights[layer] = self.weights[layer] - self.alpha * self.dW[layer]
            # self.bias[layer] = self.bias[layer] - self.alpha * self.dB[layer]
            self.weights[layer] = self.weights[layer] - self.alpha * (m_dw_bias_corrected / (np.sqrt(v_dw_bias_corrected) + EPSILON))
            self.bias[layer] = self.bias[layer] - self.alpha * (m_db_bias_corrected / (np.sqrt(v_db_bias_corrected) + EPSILON))

    def fit(self, input_vector, expected_output, epoch, batch_num):

        self.forward_propagation(input_vector)
        self.backward_propagate(input_vector, expected_output, epoch)
        loss = np.sum(np.array(self.loss_function(self.a[self.no_of_hidden_layers], expected_output)))
        self.loss_for_batches.append(loss)
        if epoch % 1000 == 0 and batch_num % 100 == 0:
            print('Loss for Epoch: ', epoch, ' is: ', loss)

        # # print('Initial Weights: ', self.weights)
        # # epoch = 1
        # loss_sum = np.sum(np.array(self.loss_function(np.zeros(np.shape(expected_output)), expected_output)))
        # while loss_sum > 10 ** -3:
        #     # print('Weights')
        #     # print('INPUT SHAPE: ', np.shape(input_vector))
        #     self.forward_propagation(input_vector)
        #     self.backward_propagate(input_vector, expected_output, epoch)
        #     loss = np.array(self.loss_function(self.a[self.no_of_hidden_layers], expected_output))
        #     # print(np.shape(loss))
        #     loss_sum = np.sum(loss)
        #     self.loss_for_epochs.append(loss_sum)
        #     # print('Weights: ', self.weights)
        #
        #     # print('Expected Output: ', expected_output)
        #     # print('Actual Output: ', self.a[self.no_of_hidden_layers])
        #     print('Loss for Epoch: ', epoch, ' is: ', loss_sum)
        #
        #     if epoch % 1000 == 0:
        #         np.save(ONE_LAYER_WEIGHTS_PATH, self.weights)
        #         # print('Expected Output: ', expected_output)
        #         # print('Actual Output: ', self.a[self.no_of_hidden_layers])
        #         print('Loss for Epoch: ', epoch, ' is: ', self.loss_for_epochs[epoch - 1])
        #     epoch += 1

    def predict(self, input_vector):
        for layer in range(self.no_of_hidden_layers + 1):
            if layer == 0:
                self.z[layer] = (self.weights[layer] @ input_vector) + self.bias[layer]
            else:
                self.z[layer] = (self.weights[layer] @ self.a[layer - 1]) + self.bias[layer]

            if layer == self.no_of_hidden_layers:
                self.a[layer] = linear(self.z[layer])
            else:
                self.a[layer] = self.activation_function(self.z[layer])

        return self.a[self.no_of_hidden_layers]

    def calculate_next_action_for_each_state(self, utility_values, predator_transitions, prey_transitions, graph):
        for agent_pos in range(len(graph)):
            for predator_pos in range(len(graph)):
                for prey_pos in range(len(graph)):
                    utility_values[agent_pos, predator_pos, prey_pos, 0] = \
                        self.calculate_next_action_for_curr_state(agent_pos, predator_pos, prey_pos, utility_values,
                                                                  predator_transitions, prey_transitions, graph)
        return utility_values

    def calculate_next_action_for_curr_state(self, agent_curr_pos, predator_curr_pos, prey_curr_pos, utility_values,
                                             predator_transitions, prey_transitions, graph):
        prey_transition_matrix = prey_transitions
        neighbours_of_agent = graph[agent_curr_pos]
        neighbours_of_predator = graph[predator_curr_pos]
        neighbours_of_prey = list(graph[prey_curr_pos])
        neighbours_of_prey.append(prey_curr_pos)

        utilities_of_all_actions = {}

        for i in range(len(neighbours_of_agent)):
            agent_possible_new_position = neighbours_of_agent[i]
            sum_for_utility = 0
            if agent_possible_new_position == prey_curr_pos:
                utilities_of_all_actions[agent_possible_new_position] = 0
            elif agent_possible_new_position == predator_curr_pos:
                utilities_of_all_actions[agent_possible_new_position] = np.inf
            else:
                predator_transition_matrix = predator_transitions[agent_possible_new_position]
                for neighbour_of_predator in neighbours_of_predator:
                    for neighbour_of_prey in neighbours_of_prey:
                        if utility_values[agent_possible_new_position, neighbour_of_predator, neighbour_of_prey, 1] == \
                                np.inf and (predator_transition_matrix[predator_curr_pos, neighbour_of_predator] == 0
                                            or prey_transition_matrix[prey_curr_pos, neighbour_of_prey] == 0):
                            sum_for_utility += 0
                        else:
                            sum_for_utility += predator_transition_matrix[predator_curr_pos, neighbour_of_predator] * \
                                   prey_transition_matrix[prey_curr_pos, neighbour_of_prey] * \
                                   utility_values[agent_possible_new_position, neighbour_of_predator, neighbour_of_prey, 1]

                utilities_of_all_actions[agent_possible_new_position] = REWARD + (BETA * sum_for_utility)

        optimal_step = min(utilities_of_all_actions, key=utilities_of_all_actions.get)
        return int(optimal_step)













