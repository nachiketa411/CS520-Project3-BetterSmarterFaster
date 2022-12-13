import random
import numpy as np


class Prey:
    def __init__(self, graph_dict):
        node_list = list(graph_dict.keys())
        self.graph = graph_dict
        self.currPos = random.choice(node_list)
        self.path = []
        self.path.append(self.currPos)

    def take_next_move(self, graph_dict):
        #print(graph_dict)
        my_neighbours = list(graph_dict[self.currPos])
        my_neighbours.append(self.currPos)
        next_move = random.choice(my_neighbours)
        self.currPos = next_move
        self.path.append(self.currPos)