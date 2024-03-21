import networkx as nx
import pygraphviz as pgv
from copy import deepcopy

UP_LIMIT = 10
DOWN_LIMIT = 10 
DEPTH_LIMIT = 15

feature_pool = ["in_degree", "out_degree", "depth", "constraint_size", "back_edge_loop", "in_branch_distance", "centrality",
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
    incoming_edge_types = dict()
    for edge in graph.in_edges(node):
        incoming_edge_types[edge] = edge_types[edge]
    # incoming_edge_types = [edge_types[edge] for edge in graph.in_edges(node)]
    return incoming_edge_types

def find_single_entry_single_exit_subgraph(graph, node):
    return True 

def merge_graph(graph, node):
    merged_graph = deepcopy(graph)

    for u, v in merged_graph.in_edges(node):
        subgraph = find_single_entry_single_exit_subgraph(merged_graph, node)

        if subgraph:
            # Merge the subgraph into node 'v'
            for n in subgraph.nodes():
                if n != node:
                    merged_graph = nx.contracted_nodes(merged_graph, node, n, self_loops=False)

    return merged_graph

def metric_ancestor(value1 , value2, alpha=0.2):
    return value1 + value2 - alpha*abs(value1 - value2)


def have_same_ancestors(graph, node, edge_types, DEPTH_LIMIT):
    # Get the predecessors of the node
    predecessors = list(graph.predecessors(node))

    # Dictionary to store the ancestors and their depths for each predecessor
    ancestors_depth = {}

    # Traverse backward from each predecessor to find their ancestors and track depths
    for pred in predecessors:
        edge_type = edge_types.get((pred, node), None)
        print("predecessor", pred)
        # Only perform the check for forward and backward edges
        if edge_type in ['Tree Edge', 'Forward Edge', 'Cross Edge']:
            ancestor_set = set()
            queue = [(pred, 0)]  # Tuple of (node, depth)

            while queue:
                current_node, depth = queue.pop(0)
                for ancestor in graph.predecessors(current_node):
                    if ancestor not in ancestor_set:
                        ancestor_set.add(ancestor)
                        queue.append((ancestor, depth + 1))

            # Store the ancestors and their depths for the current predecessor
            ancestors_depth[pred] = {ancestor: depth for ancestor, depth in queue}
            print("ancestor set", ancestor_set)
    # Find pairs of branches that share the same ancestor and calculate the metric
    num_pairs = 0
    max_metric = 0
    for ancestor in set.intersection(*[set(ancestors.keys()) for ancestors in ancestors_depth.values()]):
        depths = [ancestors_depth[pred][ancestor] for pred in ancestors_depth if ancestor in ancestors_depth[pred]]
        for i in range(len(depths)):
            for j in range(i + 1, len(depths)):
                num_pairs += 1
                branch_pair = (depths[i][0], depths[j][0])
                metric_value = metric_ancestor(depths[i], depths[j])
                max_metric = max(max_metric, metric_value)
                print(f"Branch pair {branch_pair} shares ancestor {ancestor} with metric value {metric_value}")

    return num_pairs, max_metric if num_pairs > 0 else 2 * DEPTH_LIMIT


def find_branch_pairs_with_common_ancestor(dfs_trees, node, DEPTH_LIMIT):
    # Dictionary to store the depths of the node in each DFS tree
    depths_in_trees = {}

    # Find the depth of the node in each DFS tree
    for tree, root in dfs_trees:
        if node in tree.nodes():
            depths_in_trees[root] = nx.shortest_path_length(tree, source=root, target=node)

    # Find pairs of roots (branches) that share a common ancestor
    common_ancestors = {}
    for root1, depth1 in depths_in_trees.items():
        for root2, depth2 in depths_in_trees.items():
            if root1 != root2:
                print("DFS tree", dfs_trees[0][0].edges())
                print(f"root1 {root1} root2 {root2}")
                common_ancestor = nx.lowest_common_ancestor(dfs_trees[0][0], root1, root2)
                if common_ancestor:
                    pair = tuple(sorted([root1, root2]))
                    common_ancestors[pair] = common_ancestor

    # Calculate the metric for each pair of branches with a common ancestor
    num_pairs = 0
    max_metric = 0
    for (root1, root2), ancestor in common_ancestors.items():
        depth1 = depths_in_trees[root1]
        depth2 = depths_in_trees[root2]
        metric_value = metric_ancestor(depth1, depth2)
        max_metric = max(max_metric, metric_value)
        num_pairs += 1
        print(f"Branches {root1} and {root2} share common ancestor {ancestor} with metric value {metric_value}")

    return num_pairs, max_metric if num_pairs > 0 else 2 * DEPTH_LIMIT

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

    num_pairs, max_metric = find_branch_pairs_with_common_ancestor(dfs_trees, node, DEPTH_LIMIT)
    print(f"Number of pairs of branches sharing the same ancestor: {num_pairs}")
    print(f"Biggest metric value of those branches: {max_metric}")