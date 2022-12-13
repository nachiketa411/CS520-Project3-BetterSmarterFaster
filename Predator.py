
import random
import copy

from Agent import Agent
from BiBFS import BidirectionalSearch


class Predator:
    def __init__(self, graph_dict, node_distances):
        node_list = list(graph_dict.keys())
        self.graph = graph_dict
        self.currPos = random.choice(node_list)
        self.path = []
        self.path.append(self.currPos)
        self.agent = None
        self.node_distances = node_distances

    def initialize(self, agent: Agent):
        self.agent = agent

    def take_next_move(self):
        g = copy.deepcopy(self.graph)
        graph_traverse = BidirectionalSearch(g)
        x = self.currPos
        y = self.agent.currPos
        next_move = graph_traverse.bidirectional_search(x, y)
        if len(next_move) > 1:
            self.currPos = next_move[1]
        else:
            print("Error")
        self.path.append(self.currPos)
