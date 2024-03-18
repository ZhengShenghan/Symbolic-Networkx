import networkx as nx
import pygraphviz as pgv

UP_LIMIT = 10
DOWN_LIMIT = 10 

feature_pool = ["in_degree", "out_degree", "depth", "constraint_size", "back_edge_loop", "in_branch_distance", 
                "stack depth", "constraint size", "number of symbolics", "instruction count", "I/O Interactions", "coveredLines"]

def heuristic(G, node):
    incoming_edges = G.in_edges(node)
    return True

def count_reachable_nodes(graph, start_node):
    visited = set()

    def dfs(node):
        if node not in visited:
            visited.add(node)
            for neighbor in graph.neighbors(node):
                dfs(neighbor)

    dfs(start_node)
    return len(visited)

def count_predecessors(graph, node):
    return len(list(graph.predecessors(node)))

def find_source_nodes(graph):
    return [node for node in graph.nodes if graph.in_degree(node) == 0]

def dfs_tree(graph, start):
    tree = nx.DiGraph()
    visited = set()

    def dfs(node):
        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                tree.add_edge(node, neighbor)
                dfs(neighbor)

    dfs(start)
    return tree

def get_dfs_trees(graph):
    source_nodes = find_source_nodes(graph)
    dfs_trees = [(dfs_tree(graph, node),node) for node in source_nodes]
    return dfs_trees

def depth(graph, node, source):
    depths = nx.single_source_shortest_path_length(graph, source)
    return depths[node]

def edge_type(graph, edge):
    return True

def classify_edges(graph):
    # Perform DFS and record discovery and finishing times
    discovery_time = {}
    finishing_time = {}
    time = [0]

    def dfs_visit(node):
        time[0] += 1
        discovery_time[node] = time[0]
        for neighbor in graph.neighbors(node):
            if neighbor not in discovery_time:
                dfs_visit(neighbor)
        time[0] += 1
        finishing_time[node] = time[0]

    for node in graph.nodes():
        if node not in discovery_time:
            dfs_visit(node)

    # Classify edges
    edge_types = {}
    for edge in graph.edges():
        u, v = edge
        if discovery_time[u] < discovery_time[v] and finishing_time[u] > finishing_time[v]:
            edge_types[edge] = "Tree Edge"
        elif discovery_time[u] > discovery_time[v] and finishing_time[u] < finishing_time[v]:
            edge_types[edge] = "Back Edge"
        elif discovery_time[u] < discovery_time[v] and finishing_time[u] < finishing_time[v]:
            edge_types[edge] = "Forward Edge"
        else:
            edge_types[edge] = "Cross Edge"

    return edge_types

def local_degree_centrality(graph, node, radius):
    subgraph = nx.ego_graph(graph, node, radius=radius)
    return nx.degree_centrality(subgraph)[node]

def local_betweenness_centrality(graph, node, radius):
    # Create a subgraph centered at the given node with the specified radius
    subgraph = nx.ego_graph(graph, node, radius=radius)

    # If the subgraph is a multigraph, convert it to a simple graph
    if isinstance(subgraph, nx.MultiGraph) or isinstance(subgraph, nx.MultiDiGraph):
        subgraph = nx.Graph(subgraph) if not subgraph.is_directed() else nx.DiGraph(subgraph)

    return nx.betweenness_centrality(subgraph, normalized=True)[node]


def edge_type_of_node(graph, node):
    edge_types = classify_edges(graph)
    incoming_edge_types = [edge_types[edge] for edge in graph.in_edges(node)]
    return incoming_edge_types


if __name__ == '__main__':
    # Load the DOT file into an AGraph object
    agraph = pgv.AGraph("1.dot")

    # Convert the AGraph object into a NetworkX graph
    G = nx.nx_agraph.from_agraph(agraph)

    # Now you can work with the nx_graph as a regular NetworkX graph
    print(nx.info(G))

    # index maintains i.e node 104 in dot will still be 104 in adj list
    nx.write_adjlist(G, "1.adjlist")

    # 
    node = '104'
    attribute = G.nodes[node]
    in_degree = G.in_degree('104')
    out_degree = G.out_degree('104')
    print(f"In-degree of node {node}: {in_degree}")
    print(f"Out-degree of node {node}: {out_degree}")
    print(f"Attribute of node {node}: {attribute}")

    # DFS
    # dfs = list(nx.dfs_preorder_nodes(G, source = '104'))
    # print(dfs)

    # find cycle and length
    cycle = list(nx.find_cycle(G,'104',orientation="original"))
    print("cycle", cycle)
    print("cycle length", len(cycle))

    # find zero in-degree node
    # Find nodes with no indegree
    # nodes_with_no_indegree = [node for node in G.nodes() if G.in_degree(node) == 0]
    # print("zero in-degree node", nodes_with_no_indegree)


    # Get all incoming edges of the node
    incoming_edges = G.in_edges(node)

    print(f"Incoming edges of node {node}: {list(incoming_edges)}")
    # generate dfs tree for all nodes
    dfs_trees = get_dfs_trees(G)

    # Print the DFS trees
    for i, (tree,_) in enumerate(dfs_trees, start=1):
        print(f"DFS Tree {i}:")
        print(tree.edges())

    # Return depth
    max_depth = 0
    ave_depth = 0
    for (tree,source) in dfs_trees:
        if node in tree.nodes():
            node_depth = depth(tree, node, source)
            ave_depth += node_depth
            if node_depth > max_depth:
                max_depth = node_depth
            print(f"depth in dfs tree {node_depth}")
        else:
            print("not in this tree")
    print(f"max depth is {max_depth}")
    ave_depth /= len(dfs_trees)
    print(f"average depth is {ave_depth}")


    # classify edge types
    incoming_edge_types = edge_type_of_node(G, node)
    print(f"Incoming edge types for node {node}: {incoming_edge_types}")

    # count predecessors
    num_predecessors = count_predecessors(G, node)
    print(f"Number of predecessors for node 4: {num_predecessors}")

    # count reachable nodes
    num_reachable_nodes = count_reachable_nodes(G, node)
    print(f"Number of nodes reachable from node 2: {num_reachable_nodes}")

    local_degree = local_degree_centrality(G, node, UP_LIMIT)
    local_betweenness = local_betweenness_centrality(G, node, UP_LIMIT)

    print(f"Local degree centrality of node {node} within radius {UP_LIMIT}: {local_degree}")
    print(f"Local betweenness centrality of node {node} within radius {UP_LIMIT}: {local_betweenness}")