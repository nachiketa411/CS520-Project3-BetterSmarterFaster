import copy
import random

import numpy as np

from Constants import REWARD, BETA, NO_OF_TIMES_I_WANT_UTILITIES_TO_BE_CONSISTENT


class ValueIteration:

    # A state is defined as [Agent_pos, Predator_pos, Prey_pos]

    # Q: How many distinct states are possible in this environment?
    # A: Given how each player can move to 50 different positions, the total number of distinct states possible are:
    # 50 x 50 x 50 i.e. 125000

    # What states are easy to determine U* for?
    # A:
    # 1. When Agent and Prey are at same position: U* = 0
    # 2. When Agent and Predator are at same position: U* = infinity
    # 3. When Agent is 1 step away from Prey: U* = 1

    def __init__(self, predator_transition, prey_transition, graph_dict, node_distances):
        self.graph = graph_dict
        self.node_distances = node_distances
        # Agent, Predator, Prey, 0 -> the optimal step that should be taken
        # Agent, Predator, Prey, 1 -> the optimal utility value for the step taken
        self.utility = np.zeros((50, 50, 50, 2))
        self.utility_sum = np.sum(self.utility)
        self.predator_transition_matrix = copy.deepcopy(predator_transition)
        self.prey_transition_matrix = copy.deepcopy(prey_transition)
        self.initialize()
        self.utility_sum = np.sum(self.utility[self.utility != np.inf])

    def initialize(self):
        for agent_pos in range(len(self.graph)):
            for predator_pos in range(len(self.graph)):
                for prey_pos in range(len(self.graph)):
                    if agent_pos != prey_pos and agent_pos == predator_pos:
                        self.utility[agent_pos, predator_pos, prey_pos, 0] = agent_pos
                        self.utility[agent_pos, predator_pos, prey_pos, 1] = np.inf
                    if agent_pos != prey_pos and agent_pos != predator_pos:
                        # initialised randomly in the range of 25 since that is the maximum shortest distance 2 players
                        # can have.
                        self.utility[agent_pos, predator_pos, prey_pos, 0] = random.randint(0, 49)
                        self.utility[agent_pos, predator_pos, prey_pos, 1] = random.randint(0, 25)
                    if self.node_distances[agent_pos][prey_pos] == 1 and agent_pos != predator_pos:
                        self.utility[agent_pos, predator_pos, prey_pos, 0] = prey_pos
                        self.utility[agent_pos, predator_pos, prey_pos, 1] = 1

    # U(agent, predator, prey) = ?
    def calc_utility_of_current_state(self, agent_curr_pos, predator_curr_pos, prey_curr_pos):

        if agent_curr_pos == predator_curr_pos or agent_curr_pos == prey_curr_pos \
                or agent_curr_pos in self.graph[prey_curr_pos]:
            return False

        prey_transition_matrix = self.prey_transition_matrix
        # All possible actions of the agent is basically movement to one of its neighbours
        # We will be calculating the new value of U(agent_curr_pos, pred_curr_pos, prey_curr_pos) using the previous
        # value of the utility
        neighbours_of_agent = self.graph[agent_curr_pos]
        neighbours_of_predator = self.graph[predator_curr_pos]
        neighbours_of_prey = list(self.graph[prey_curr_pos])
        neighbours_of_prey.append(prey_curr_pos)

        # utilities_of_all_actions = [0] * len(neighbours_of_agent)
        utilities_of_all_actions = {}

        for i in range(len(neighbours_of_agent)):
            agent_possible_new_position = neighbours_of_agent[i]
            sum_for_utility = 0
            if agent_possible_new_position == prey_curr_pos:
                # utilities_of_all_actions[i] = 0
                utilities_of_all_actions[agent_possible_new_position] = 0
            elif agent_possible_new_position == predator_curr_pos:
                # utilities_of_all_actions[i] = np.inf
                utilities_of_all_actions[agent_possible_new_position] = np.inf
            else:
                predator_transition_matrix = self.predator_transition_matrix[agent_possible_new_position]
                for neighbour_of_predator in neighbours_of_predator:
                    for neighbour_of_prey in neighbours_of_prey:
                        if self.utility[agent_possible_new_position, neighbour_of_predator, neighbour_of_prey, 1] == \
                                np.inf and (predator_transition_matrix[predator_curr_pos, neighbour_of_predator] == 0
                                            or prey_transition_matrix[prey_curr_pos, neighbour_of_prey] == 0):
                            sum_for_utility += 0
                        else:
                            sum_for_utility += predator_transition_matrix[predator_curr_pos, neighbour_of_predator] * \
                                   prey_transition_matrix[prey_curr_pos, neighbour_of_prey] * \
                                   self.utility[agent_possible_new_position, neighbour_of_predator, neighbour_of_prey, 1]

                # utilities_of_all_actions[i] = REWARD + (BETA * sum_for_utility)
                utilities_of_all_actions[agent_possible_new_position] = REWARD + (BETA * sum_for_utility)

        new_optimal_step = min(utilities_of_all_actions, key=utilities_of_all_actions.get)
        prev_optimal_step = self.utility[agent_curr_pos, predator_curr_pos, prey_curr_pos, 0]
        # 0 -> the optimal step that should be taken
        # 1 -> the optimal utility value for the step taken
        self.utility[agent_curr_pos, predator_curr_pos, prey_curr_pos, 0] = new_optimal_step
        self.utility[agent_curr_pos, predator_curr_pos, prey_curr_pos, 1] = utilities_of_all_actions[new_optimal_step]

        if new_optimal_step != prev_optimal_step:
            return True
        return False

    def value_iteration(self):
        curr_sum = 0
        count = 0
        print('Initial Utility Sum: ', np.sum(self.utility[self.utility != np.inf]))
        # print('Length of Graph: ', len(self.graph))
        # while abs(self.utility_sum - curr_sum) >= (10 ** -5):
        continue_loop = True
        count_of_sum = 0
        while continue_loop:
            sum_of_change_in_optimal_step = 0
            count += 1
            curr_sum = np.sum(self.utility[self.utility != np.inf])
            for agent_pos in range(len(self.graph)):
                # print('Agent: --------------', agent_pos)
                for predator_pos in range(len(self.graph)):
                    # print('Predator: --------------', predator_pos)
                    for prey_pos in range(len(self.graph)):
                        # print('Prey: --------------', prey_pos)
                        if self.calc_utility_of_current_state(agent_pos, predator_pos, prey_pos):
                            sum_of_change_in_optimal_step += 1

            # print('No. of changes: ', sum_of_change_in_optimal_step)

            # We are checking if there is a change in the optimal next step
            # We hope to at least have 2 consecutive iterations where there are no changes in the next optimal step
            if sum_of_change_in_optimal_step == 0:
                count_of_sum += 1
            else:
                count_of_sum = 0
            if count_of_sum > NO_OF_TIMES_I_WANT_UTILITIES_TO_BE_CONSISTENT - 1:
                continue_loop = False

        print('Final Utility Sum: ', np.sum(self.utility[self.utility != np.inf]))
        # print('Diff: ', abs(self.utility_sum - curr_sum))
        # print('Count: ', count)
