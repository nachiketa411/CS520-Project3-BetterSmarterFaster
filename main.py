# Press the green button in the gutter to run the script.
import copy
import json

import numpy as np

from ActivationFunction import sigmoid, gradient_sigmoid
from AgentUStar import AgentUStar
from AgentUpartial import AgentUpartial
from Constants import ENVIRONMENT_PATH, GRAPH_DIST_PATH, UTILITIES_PATH, BATCH_SIZE, NO_OF_EPOCHS, INFINITY
from LossFunctions import euclidean_loss, gradient_euclidean_loss
from NeuralNetwork import NeuralNetwork
from Predator import Predator
from Prey import Prey
from TransitionMatrix import TransitionMatrix
from ValueIteration import ValueIteration


def read_from_json():
    graph_dictionary = {}
    distances_dictionary = {}

    with open(ENVIRONMENT_PATH, "r") as env_file:
        json_object = json.load(env_file)

    with open(GRAPH_DIST_PATH, "r") as node_dist:
        json_object_2 = json.load(node_dist)

    for g_id in json_object.items():
        graph_id = int(g_id[0])
        graph = {}
        for k in g_id[1]:
            graph[int(k)] = g_id[1][k]
        graph_dictionary[graph_id] = graph

    for g_id in json_object_2.items():
        graph_id = int(g_id[0])
        distance_mat = []
        for j in g_id[1].items():
            temp = []
            for k in j[1].items():
                temp.append(int(k[1]))
            distance_mat.append(temp)
        distances_dictionary[graph_id] = distance_mat
    return graph_dictionary, distances_dictionary


# def set_training_testing_data(dataset, batch_size):
#     # dataset = (m x n)
#     # dataset.shape[0] = m i.e. No. of rows or input_size
#     total_indices = set(range(dataset.shape[0]))
#     training_indices = set(random.sample(total_indices, batch_size))
#     testing_indices = list(total_indices - training_indices)
#     training_indices = list(training_indices)
#
#     np.random.shuffle(training_indices)
#     np.random.shuffle(testing_indices)
#
#     training_data = dataset[training_indices]
#     testing_data = dataset[testing_indices]

def input_x_y(node_distances, utility, NO_OF_FEATURES, BATCH_SIZE):
    x_values = np.zeros((NO_OF_FEATURES, BATCH_SIZE))
    y_values = np.zeros((BATCH_SIZE, 1))
    agent_low = 0
    agent_high = 2500
    pred_low = 0
    pred_high = 50
    for agent_pos in range(50):
        for i in range(agent_low, agent_high):
            x_values[0, i] = agent_pos
        for pred_pos in range(50):
            for j in range(pred_low, pred_high):
                x_values[1, j] = pred_pos
            for prey_pos in range(50):
                z = pred_low + prey_pos
                x_values[2, z] = prey_pos
            pred_low += 50
            pred_high += 50
        agent_low += 2500
        agent_high += 2500
    for column in range(BATCH_SIZE):
        agent_loc = int(x_values[0, column])
        predator_loc = int(x_values[1, column])
        prey_loc = int(x_values[2, column])
        x_values[3, column] = node_distances[agent_loc][prey_loc]
        x_values[4, column] = node_distances[agent_loc][predator_loc]
        x_values[5, column] = node_distances[prey_loc][predator_loc]
        y_values[column] = utility[agent_loc, predator_loc, prey_loc, 1]
        y_values[y_values == np.inf] = INFINITY
    return [x_values, y_values.T]


def shuffle_and_split(NO_OF_PARTS, x_values, utilities, BATCH_SIZE):
    shuffled_x = copy.deepcopy(x_values)
    shuffled_y = np.zeros((BATCH_SIZE, 1))
    np.random.shuffle(np.transpose(shuffled_x))
    for column in range(BATCH_SIZE):
        agent_loc = int(shuffled_x[0][column])
        predator_loc = int(shuffled_x[1][column])
        prey_loc = int(shuffled_x[2][column])
        shuffled_y[column] = utilities[agent_loc, predator_loc, prey_loc, 1]
    shuffled_y = shuffled_y.T
    shuffled_split_x = np.hsplit(shuffled_x, NO_OF_PARTS)
    shuffled_split_y = np.hsplit(shuffled_y, NO_OF_PARTS)
    shuffled_split_y[shuffled_split_y == np.inf] = INFINITY
    return [shuffled_split_x, shuffled_split_y]


if __name__ == '__main__':
    # ---------------------Read from JSON-------------------
    converted_graph, converted_distances = read_from_json()
    # ----------------------------------------------------
    # Calc
    # ----------------------------------------------------
    utility_values_for_each_graph = {}
    success_of_Agent = 0
    failure_rate_1 = 0
    failure_rate_2 = 0

    my_graph_utilities = np.load(UTILITIES_PATH, allow_pickle=True)

    arr = input_x_y(converted_distances[0], my_graph_utilities[()][0], 6, 125000)
    split_arr = shuffle_and_split(10, arr[0], my_graph_utilities[()][0], 125000)
    # print(arr[0])
    # print(np.shape(arr[0]))
    # print(split_arr[0][0], split_arr[1])
    # print(np.shape(split_arr[0]))

    for k in range(1):
        nn = NeuralNetwork(6, sigmoid, gradient_sigmoid,
                           euclidean_loss, gradient_euclidean_loss, BATCH_SIZE, NO_OF_EPOCHS)
        nn.fit(arr[0], arr[1])
    # for k in range(1):
    #     # # Since the transition matrices calculated here won't change for different iterations of the same graph,
    #     # # we precalculate them and use them for each of the 30 iteration
    #     transitions = TransitionMatrix(converted_graph[k])
    #     # predator_transitions = transitions.predator_transition
    #     prey_transitions = transitions.prey_transition
    #     #
    #     # # Calculate the Utility of each state
    #     # valueIterator = ValueIteration(predator_transitions, prey_transitions,
    #     #                                converted_graph[k], converted_distances[k])
    #     # valueIterator.value_iteration()
    #     # utility_values_for_each_graph[k] = valueIterator.utility
    #     utility_values_for_each_graph[k] = my_graph_utilities[()][k]
    #
    #     for i in range(1):
    #         prey = Prey(converted_graph[k])
    #         predator = Predator(converted_graph[k], converted_distances[k])
    #         agent = AgentUpartial(prey, converted_graph[k])
    #         agent.initialize(predator)
    #         predator.initialize(agent)
    #         agent.set_utility(utility_values_for_each_graph[k])
    #
    #         while agent.utility[agent.currPos, predator.currPos, prey.currPos, 1] == np.inf:
    #             prey = Prey(converted_graph[k])
    #             predator = Predator(converted_graph[k], converted_distances[k])
    #             # agent = AgentUStar(prey, converted_graph[k])
    #             agent = AgentUpartial(prey, converted_graph[k])
    #             agent.initialize(predator)
    #             predator.initialize(agent)
    #             agent.set_utility(utility_values_for_each_graph[k])
    #
    #         # steps_taken = agent.move_agent()
    #         steps_taken = agent.move_agent(prey_transitions)
    #         if steps_taken[1] == -1:
    #             success_of_Agent += 1
    #         if steps_taken[1] == -2:
    #             failure_rate_1 += 1
    #         if steps_taken[1] == -3:
    #             failure_rate_2 += 1
    #
    # # np.save(UTILITIES_PATH, utility_values_for_each_graph)
    # print('Total Number of Successes: ', success_of_Agent)
    # print('Total Number of Deaths   : ', failure_rate_1)
    # print('Total Number of Hangs    : ', failure_rate_2)
    # print("Agent path: ", agent.path)
    # print("Prey path: ", prey.path)
    # print("Predator path: ", predator.path)

    # To recall the data from the npy file:
    # my_graph_utilities = np.load(UTILITIES_PATH)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
