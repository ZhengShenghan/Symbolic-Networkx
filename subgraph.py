import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import pygraphviz as pgv

# Load the DOT file into an AGraph object
agraph = pgv.AGraph("1.dot")

# Convert the AGraph object into a NetworkX graph
G = nx.nx_agraph.from_agraph(agraph)

node = '104'

G = nx.ego_graph(G, node, radius=10)
# Specify the path for the DOT file
dot_file_path = '1_sub.dot'

# Save the graph as a DOT file
write_dot(G, dot_file_path)