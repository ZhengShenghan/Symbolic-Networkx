import networkx as nx
import pygraphviz as pgv

UP_LIMIT = 10
DOWN_LIMIT = 10 

def heuristic(graph, node):
    return True

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
    dfs_trees = [dfs_tree(graph, node) for node in source_nodes]
    return dfs_trees

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
    dfs = list(nx.dfs_preorder_nodes(G, source = '104'))
    print(dfs)

    # find cycle
    cycle = list(nx.find_cycle(G,'104',orientation="original"))
    print(cycle)

    # find zero in-degree node
    # Find nodes with no indegree
    nodes_with_no_indegree = [node for node in G.nodes() if G.in_degree(node) == 0]
    print("zero in-degree node", nodes_with_no_indegree)

    # generate dfs tree for all nodes
    dfs_trees = get_dfs_trees(G)
    # Print the DFS trees
    for i, tree in enumerate(dfs_trees, start=1):
        print(f"DFS Tree {i}:")
        print(tree.edges())