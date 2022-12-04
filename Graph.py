import random
from queue import Queue

import networkx as nx

from Constants import NO_OF_NODES


class Graph:
    def __init__(self):
        self.graph = {}
        self.distance_from_each_node = {}
        self.generate_graph(NO_OF_NODES)
        self.calculate_distance_from_each_node()

    def generate_graph(self, no_of_nodes):
        for i in range(no_of_nodes):
            # this will create a circular graph
            if i == 0:
                self.graph[i] = [no_of_nodes - 1, i + 1]
            elif i == no_of_nodes - 1:
                self.graph[i] = [i - 1, 0]
            else:
                self.graph[i] = [i - 1 % no_of_nodes, i + 1 % no_of_nodes]
        # we need to add the remaining edges according to the conditions given for the environment
        set_of_nodes_with_degree_2 = set(self.graph.keys())

        while len(set_of_nodes_with_degree_2) > 0:
            random_node_choice = random.choice(list(set_of_nodes_with_degree_2))
            list_of_possible_connections = []
            for i in range(2, 6):
                list_of_possible_connections.append((random_node_choice - i) % no_of_nodes)
                list_of_possible_connections.append((random_node_choice + i) % no_of_nodes)

            filtered_list = list(filter(lambda a: a in set_of_nodes_with_degree_2, list_of_possible_connections))

            if len(filtered_list) == 0:
                set_of_nodes_with_degree_2.remove(random_node_choice)
            else:
                random_node_connection_choice = random.choice(filtered_list)
                self.graph[random_node_choice].append(random_node_connection_choice)
                self.graph[random_node_connection_choice].append(random_node_choice)
                set_of_nodes_with_degree_2.remove(random_node_choice)
                set_of_nodes_with_degree_2.remove(random_node_connection_choice)

        return self.graph

    def calculate_distance_from_each_node(self):
        for i in (self.graph.keys()):
            self.distance_from_each_node[i] = self.calculate_distance(i)

    def calculate_distance(self, source):
        Q = Queue()
        distance = {k: 9999999 for k in self.graph.keys()}
        visited_vertices = set()
        Q.put(source)
        while not Q.empty():
            vertex = Q.get()
            if vertex == source:
                distance[vertex] = 0
            for u in self.graph[vertex]:
                if u not in visited_vertices:
                    # update the distance
                    if distance[u] > distance[vertex] + 1:
                        distance[u] = distance[vertex] + 1
                    Q.put(u)
                    visited_vertices.update({u})
        return distance
