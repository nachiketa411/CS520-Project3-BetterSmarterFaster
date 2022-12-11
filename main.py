# Press the green button in the gutter to run the script.
import json

import numpy as np

from AgentUStar import AgentUStar
from AgentUpartial import AgentUpartial
from Constants import ENVIRONMENT_PATH, GRAPH_DIST_PATH, UTILITIES_PATH
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


def input_x_y(node_distances, utility):
    x_values = [[0] * 125000] * 6
    y_values = [0] * 125000
    agent_low = 0
    agent_high = 2500
    prey_low = 0
    prey_high = 50
    for agent_pos in range(50):
        for i in range(agent_low, agent_high):
            x_values[0][i] = agent_pos
        for prey_pos in range(50):
            for j in range(prey_low, prey_high):
                x_values[1][j] = prey_pos
            for predator_pos in range(50):
                x_values[2][prey_low + predator_pos] = predator_pos
            prey_low += 50
            prey_high += 50
        agent_low += 2500
        agent_high += 2500
    for column in range(125000):
        agent_loc = x_values[0][column]
        prey_loc = x_values[1][column]
        predator_loc = x_values[2][column]
        x_values[3][column] = node_distances[agent_loc][prey_loc]
        x_values[4][column] = node_distances[agent_loc][predator_loc]
        x_values[5][column] = node_distances[prey_loc][predator_loc]
        y_values[column] = utility[agent_loc, predator_loc, prey_loc, 1]

    return x_values, y_values


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

    for k in range(1):
        # # Since the transition matrices calculated here won't change for different iterations of the same graph,
        # # we precalculate them and use them for each of the 30 iteration
        transitions = TransitionMatrix(converted_graph[k])
        # predator_transitions = transitions.predator_transition
        prey_transitions = transitions.prey_transition
        #
        # # Calculate the Utility of each state
        # valueIterator = ValueIteration(predator_transitions, prey_transitions,
        #                                converted_graph[k], converted_distances[k])
        # valueIterator.value_iteration()
        # utility_values_for_each_graph[k] = valueIterator.utility
        utility_values_for_each_graph[k] = my_graph_utilities[()][k]

        for i in range(1):
            prey = Prey(converted_graph[k])
            predator = Predator(converted_graph[k], converted_distances[k])
            agent = AgentUpartial(prey, converted_graph[k])
            agent.initialize(predator)
            predator.initialize(agent)
            agent.set_utility(utility_values_for_each_graph[k])

            while agent.utility[agent.currPos, predator.currPos, prey.currPos, 1] == np.inf:
                prey = Prey(converted_graph[k])
                predator = Predator(converted_graph[k], converted_distances[k])
                # agent = AgentUStar(prey, converted_graph[k])
                agent = AgentUpartial(prey, converted_graph[k])
                agent.initialize(predator)
                predator.initialize(agent)
                agent.set_utility(utility_values_for_each_graph[k])

            # steps_taken = agent.move_agent()
            steps_taken = agent.move_agent(prey_transitions)
            if steps_taken[1] == -1:
                success_of_Agent += 1
            if steps_taken[1] == -2:
                failure_rate_1 += 1
            if steps_taken[1] == -3:
                failure_rate_2 += 1

    # np.save(UTILITIES_PATH, utility_values_for_each_graph)
    print('Total Number of Successes: ', success_of_Agent)
    print('Total Number of Deaths   : ', failure_rate_1)
    print('Total Number of Hangs    : ', failure_rate_2)
    print("Agent path: ", agent.path)
    print("Prey path: ", prey.path)
    print("Predator path: ", predator.path)

    # To recall the data from the npy file:
    # my_graph_utilities = np.load(UTILITIES_PATH)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
