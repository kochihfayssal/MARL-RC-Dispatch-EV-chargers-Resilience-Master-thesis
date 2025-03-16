import numpy as np
import random 

class Transport_Network :
    def __init__(self, distances, travel_times, network_size):
        self.network_size = network_size
        self.coefficients = (0.15, 4) # alpha and beta
        self.nodes = self._get_nodes()
        self.edges = self._get_edges()
        self.distances = self._get_data(distances)
        #self.traffic_volumes = traffic_volumes #list of length 24, giving the traffic volume at each time step 
        self.travel_times = self._get_data(travel_times)
    
    def _get_nodes(self):
        return [(i,j) for i in range(self.network_size) for j in range(self.network_size)]

    def _get_edges(self):
        E = []
        for i in range(self.network_size):
            for j in range(self.network_size):
                if i < self.network_size-1:
                    E.append(((i,j), (i+1,j)))
                if j < self.network_size-1:
                    E.append(((i,j), (i, j+1)))
        return E

    def _get_data(self, data):
        output = {}
        for i, edge in enumerate(self.edges):
            output[edge] = data[i]
        return output
    
    def get_distance_edge(self, node1, node2):
        edge = tuple(sorted((node1, node2)))
        return self.distances.get(edge, None)

    def get_travel_time_edge(self, node1, node2):
        edge = tuple(sorted((node1, node2)))
        return self.travel_times.get(edge, None)
    
    def get_travel_time_node(self, node):
        """
        returns the travel times of the different roads attached to a specific node.
        if a node has less than 4 roads linked to it, it assigns 0 to the missing roads. 
        the indexes refer to the following order : up, down, left, right
        """
        surrounding_times = np.zeros(4) # left, right, up, down 
        i,j = node
        surrounding_times[0] = self.get_travel_time_edge(node, (i,j-1)) if j > 0 else 0
        surrounding_times[1] = self.get_travel_time_edge(node, (i,j+1)) if j < self.network_size-1 else 0
        surrounding_times[2] = self.get_travel_time_edge(node, (i-1,j)) if i > 0 else 0
        surrounding_times[3] = self.get_travel_time_edge(node, (i+1,j)) if i < self.network_size-1 else 0
        return surrounding_times 

