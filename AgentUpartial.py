import copy
import random
import numpy as np
from Agent import Agent
from Constants import NO_OF_STEPS_4, NO_OF_NODES


class AgentUpartial(Agent):
    def move_agent(self, prey_transition_matrix):
        count = 0
        belief = [1 / 49] * 50
        belief[self.currPos] = 0
        while count <= NO_OF_STEPS_4:
            # print(count)
            # Survey the node with the highest Probability
            to_survey = self.select_node(belief)
            # print("Before survey",sum(belief), to_survey)
            # print(belief)
            belief = self.update_belief(belief, to_survey)
            # print("After survey", sum(belief))
            # print(belief)
            next_move = self.get_next_move(belief)

            self.currPos = next_move
            self.path.append(next_move)
            if self.currPos == self.prey.currPos:
                print("Yippiieeee")
                count += 1
                return [count, -1, self.counter_for_prey_actually_found, self.counter_for_predator_actually_found]
            elif self.currPos == self.predator.currPos:
                print("Ded")
                count += 1
                return [count, -2, self.counter_for_prey_actually_found, self.counter_for_predator_actually_found]

            belief = self.update_belief(belief, self.currPos)
            # print("After Agent moves", sum(belief), next_move)
            # print(belief)
            self.prey.take_next_move(copy.deepcopy(self.graph))
            if self.currPos == self.prey.currPos:
                print("Yippiieeee")
                count += 1
                return [count, -1, self.counter_for_prey_actually_found, self.counter_for_predator_actually_found]

            belief = self.update_belief_using_transition_mat(belief, prey_transition_matrix)
            # print("After Prey moves", sum(belief))
            # print(belief)
            self.predator.take_next_move()
            if self.currPos == self.predator.currPos:
                print("Ded")
                count += 1
                return [count, -2, self.counter_for_prey_actually_found, self.counter_for_predator_actually_found]
            count += 1
        return [count, -3, self.counter_for_prey_actually_found, self.counter_for_predator_actually_found]

    def get_next_move(self, belief):
        upartial = {}
        for possible_next_move in self.graph[self.currPos]:
            summation = 0
            for i in range(len(belief)):
                if self.utility[possible_next_move, self.predator.currPos, i, 1] == np.inf:
                    summation = np.inf
                    break
                else:
                    summation += (belief[i] * self.utility[possible_next_move, self.predator.currPos, i, 1])
            upartial[possible_next_move] = summation
        optimal_step = min(upartial, key=upartial.get)
        # optimal_step = self.utility[self.currPos, self.predator.currPos, possible_prey, 0]
        return int(optimal_step)

    def select_node(self, belief_mat):
        max_in_belief_mat = max(belief_mat)
        possible_nodes = []
        for i in range(len(belief_mat)):
            if belief_mat[i] == max_in_belief_mat:
                possible_nodes.append(i)
        return random.choice(possible_nodes)

    def update_belief(self, belief_mat, node):
        if node == self.prey.currPos:
            belief_mat = [0] * NO_OF_NODES
            belief_mat[node] = 1
        else:
            temp = 1 - belief_mat[node]
            belief_mat[node] = 0
            for i in range(len(belief_mat)):
                belief_mat[i] = belief_mat[i] / temp
        return belief_mat

    def update_belief_using_transition_mat(self, belief_mat, transition_mat):
        new_belief_mat = [0] * NO_OF_NODES
        for i in range(len(belief_mat)):
            # P(Prey in i)= Summation(P(Prey in neighbour of i)*P(Prey in neighbour of i|Prey in i))
            summation = 0
            for j in range(len(transition_mat[i])):
                summation += (belief_mat[j] * transition_mat[j][i])
            new_belief_mat[i] = summation
        return new_belief_mat
