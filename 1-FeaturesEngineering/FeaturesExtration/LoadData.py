import networkx as nx
import numpy as np

disconnected = ['25769', '21986', '33165', '33100', '30874', 
                '27534', '30556', '25156', '10049', '20596', 
                '33175', '30504', '17120', '26905', '30766', 
                '28947', '27435', '31157', '25601', '22507', 
                '31377', '21465', '32864', '25516', '12990', 
                '31785', '30660', '28860', '25499', '30697', 
                '28944', '27553', '28372', '5703', '23525', 
                '2015', '23438', '28360', '26210', '28936', 
                '28948', '28061', '23709', '28823', '33221', 
                '32447', '26897', '30474', '25283', '29048', 
                '33213', '27520', '24663', '32806', '31081', 
                '33113', '31140', '30959', '29280', '33199', 
                '31007', '32605', '31995', '30829']

def load():
    # Load training edges and build graph
    training_set = []
    G = nx.DiGraph()
    with open("../../data/training.txt", "r") as f:
        for line in f:
            line = list(map(int,line.split()))
            training_set.append(line)
            G.add_nodes_from(line[:2])
            if line[2]==1:
                G.add_edge(line[0], line[1])
            
    training_set = np.array(training_set)

    # Load testing Edges
    testing_set = []
    with open("../../data/testing.txt", "r") as f:
        for line in f:
            line = list(map(int,line.split()))
            testing_set.append(line)
            
    testing_set = np.array(testing_set)

    return G, training_set, testing_set, disconnected