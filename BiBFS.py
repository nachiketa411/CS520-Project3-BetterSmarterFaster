class BidirectionalSearch:

    def __init__(self,graph):

        self.graph=graph

        #Creating Queues for Source and Destination Nodes
        self.src_queue = list()
        self.dest_queue = list()

        #Initialising visited queues for Source and Destination as False
        self.src_visited = [False] * len(self.graph)
        self.dest_visited = [False] * len(self.graph)

        #Initialising Source and Destination Parent Nodes
        self.src_parent = [None] * len(self.graph)
        self.dest_parent = [None] * len(self.graph)

    # Function for Breadth First Search
    def bfs(self, direction = 'forward'):

        if direction == 'forward':

            # BFS in forward direction
            current = self.src_queue.pop(0)
            connected_node = self.graph[current]
            #print(connected_node)
            while connected_node:
                vertex = connected_node.pop(0)
                #print(vertex)
                if not self.src_visited[vertex]:
                    self.src_queue.append(vertex)
                    self.src_visited[vertex] = True
                    self.src_parent[vertex] = current
        else:

            # BFS in backward direction
            current = self.dest_queue.pop(0)
            connected_node = self.graph[current]
            #print(connected_node)
            while connected_node:
                vertex = connected_node.pop(0)
                #print(vertex)
                if not self.dest_visited[vertex]:
                    self.dest_queue.append(vertex)
                    self.dest_visited[vertex] = True
                    self.dest_parent[vertex] = current


    # Check for intersecting vertex
    def is_intersecting(self):

        # Returns intersecting node
        # if present else -1
        for i in self.graph.keys():
            if (self.src_visited[i] and
                    self.dest_visited[i]):
                return i

        return -1

    # Print the path from source to target
    def print_path(self, intersecting_node, src, dest):
        # Print final path from source to destination
        path = list()
        path.append(intersecting_node)
        i = intersecting_node

        while i != src:
            path.append(self.src_parent[i])
            i = self.src_parent[i]

        path = path[::-1]
        i = intersecting_node
        while i != dest:
            path.append(self.dest_parent[i])
            i = self.dest_parent[i]
        return path

    # Function for bidirectional searching
    def bidirectional_search(self, src, dest):
        # Add source to queue and mark visited as True and add its parent as -1
        #print(dest)
        self.src_queue.append(src)
        self.src_visited[src] = True
        self.src_parent[src] = -1

        # Add destination to queue and mark visited as True and add its parent as -1
        self.dest_queue.append(dest)
        self.dest_visited[dest] = True
        self.dest_parent[dest] = -1

        while self.src_queue and self.dest_queue:

            # BFS in forward direction from Source Vertex
            self.bfs(direction = 'forward')
            #print("A")
            # BFS in reverse direction from Destination Vertex
            self.bfs(direction = 'backward')
            #print(self.dest_parent)

            # Check for intersecting vertex
            intersecting_node = self.is_intersecting()

            # If intersecting vertex exists then path from source to destination exists
            if intersecting_node != -1:
                #print(f"Path exists between {src} and {dest}")
                #print(f"Intersection at : {intersecting_node}")
                path=self.print_path(intersecting_node, src, dest)
                self.src_queue.clear()
                self.src_visited.clear()
                self.src_parent.clear()
                self.dest_queue.clear()
                self.dest_visited.clear()
                self.dest_parent.clear()
                return path
        return -1
