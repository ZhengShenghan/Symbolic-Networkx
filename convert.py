import networkx as nx
import pygraphviz as pgv
from copy import deepcopy
import json
# import programl as pg
import os

def find_node_with_merge_point_block(graph):
    for node, data in graph.nodes(data=True):
        features = data['features'] # features is a string
        # exit(0)
        # text_attr = nx.get_node_attributes(graph, )
        if '<MERGE_POINT_BLOCK>' in features:
            return node
    return None




if __name__ == '__main__':
    output = dict()
    # Load the DOT file into an AGraph object
    agraph = pgv.AGraph("1.dot")

    # Convert the AGraph object into a NetworkX graph
    G = nx.nx_agraph.from_agraph(agraph)
    node = find_node_with_merge_point_block(G)

    print(f'node with merge block {node}')

    try:
        cycle = list(nx.find_cycle(G, '1', orientation="original"))
        print("cycle", cycle)
        print("cycle length", len(cycle))
        output["back edge loop length"] = len(cycle)
    except nx.NetworkXNoCycle:
        print("No cycle found.")
        output["back edge loop length"] = 0  # Set the loop length to 0 if no cycle is found

    print(output)