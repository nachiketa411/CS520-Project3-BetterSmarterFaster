import copy

import numpy as np

from BiBFS import BidirectionalSearch
from Constants import PROB_OF_DISTRACTED_PREDATOR


class TransitionMatrix:
    def __init__(self, graph_dict):
        self.graph = graph_dict
        self.prey_transition = np.zeros((50, 50))
        # Transition_Matrix(i, j, k)
        # i -> current position of Predator
        # j -> current position of Agent
        # k -> positions that predator can move towards
        self.predator_transition = np.zeros((50, 50, 50))

        self.generate_prey_transition_matrix()
        self.generate_predator_transition_matrix()

    def generate_prey_transition_matrix(self):
        for prey_pos in range(len(self.graph)):
            neighbours = self.graph[prey_pos]
            prob = 1 / (len(neighbours) + 1)
            self.prey_transition[prey_pos, prey_pos] = prob
            for neighbor in neighbours:
                self.prey_transition[prey_pos, neighbor] = prob

    # Calculate: transition_matrix(agent_pos, curr_predator_pos, neighbours_for_predator_to_go_to)
    def generate_predator_transition_matrix(self):
        for agent_pos in range(len(self.graph)):
            for predator_pos in range(len(self.graph)):
                neighbours_of_predator = self.graph[predator_pos]
                best_equivalent_set_of_moves = self.get_neighbours_with_equal_distance_to_agent(neighbours_of_predator,
                                                                                                agent_pos)
                for neighbour in neighbours_of_predator:
                    prob = 0
                    if neighbour in best_equivalent_set_of_moves:
                        prob += (1 - PROB_OF_DISTRACTED_PREDATOR) / (len(best_equivalent_set_of_moves))
                    prob += PROB_OF_DISTRACTED_PREDATOR / (len(neighbours_of_predator))
                    self.predator_transition[agent_pos, predator_pos, neighbour] = prob

    # Finds a path from the given list of neighbours to the destination/pos_y
    def find_path(self, neighbours, pos_y):
        path_dictionary = {}
        for i in range(len(neighbours)):
            temp = copy.deepcopy(self.graph)
            bi_bfs = BidirectionalSearch(temp)
            x = neighbours[i]
            y = pos_y
            path = bi_bfs.bidirectional_search(x, y)
            path_dictionary[neighbours[i]] = path
        return path_dictionary

    def get_neighbours_with_equal_distance_to_agent(self, neighbours, destination):
        dict_of_paths = self.find_path(neighbours, destination)
        list_of_paths = dict_of_paths.values()
        list_of_dist = []
        for path in list_of_paths:
            list_of_dist.append(len(path))
        smallest_dist = min(list_of_dist)
        best_neighbours = []
        for neighbour, path in dict_of_paths.items():
            if len(path) == smallest_dist:
                best_neighbours.append(neighbour)
        return best_neighbours
