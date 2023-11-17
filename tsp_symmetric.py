import pulp as pl
import math as math
import random as rnd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from datetime import datetime
from typing import Dict, List, Tuple


def distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Returns the Euclidean distance between two points on the plane.

    Parameters:
        point1 (Tuple[float, float]): The coordinates of the first point (x1, y1).
        point2 (Tuple[float, float]): The coordinates of the second point (x2, y2).

    Returns:
        float: The Euclidean distance between the two points.
    """
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])


def tsp_symmetric_linprog(init_graph: nx.Graph) -> nx.Graph:
    """
    Solves the symmetric traveling salesman problem using integer linear programming.

    Parameters:
        init_graph (nx.Graph): The graph representing the distances between points.

    Returns:
        nx.Graph: The optimized tour as an undirected graph.
    """
    # Get the list of edges in the input graph
    edges_list: List[Tuple[int, int]] = list(init_graph.edges())

    # Fill index dictionaries
    array_index: Dict[int, Tuple[int, int]] = {}
    array_key: Dict[Tuple[int, int], int] = {}
    for i, val in enumerate(edges_list):
        key = tuple(sorted(val))
        array_key[key] = i
        array_index[i] = key

    # Declare the model
    model = pl.LpProblem(name="tsp", sense=pl.LpMinimize)

    # Connect the solver
    solver = pl.PULP_CBC_CMD(msg=False)

    # Declare model variables
    x = [pl.LpVariable(name=f'x{i:06}', cat='Binary') for i in range(len(edges_list))]

    # Input the objective function
    model += pl.lpSum([init_graph[array_index[i][0]][array_index[i][1]]['weight'] * x[i] for i in array_index])

    # Set initial constraints for the model
    for i in range(len(init_graph)):
        model += pl.lpSum([x[array_key[key]] for key in array_key if i in key]) == 2

    step = 0
    while True:
        # Find a solution
        status = model.solve(solver)
        step += 1

        if status != 1:
            return nx.Graph()

        # Save solver result as a graph
        graph_result = nx.Graph()
        for i, val in enumerate(model.variables()):
            if val.value() == 1:
                graph_result.add_edge(*array_index[i])

        # Split the result graph into subsets
        result_sets = list(nx.connected_components(graph_result))

        print(f'Step: {step}; Length: {round(model.objective.value(), 6)}')

        # Solution found if there is only one subset in the graph
        if len(result_sets) == 1:
            return graph_result

        # For each subset, add a constraint connecting it to other subsets
        for val in result_sets:
            # Choose the algorithm for creating a constraint that involves fewer variables
            alg_1 = sum(1 for key in init_graph.edges(val) if key[0] in val and key[1] in val)
            alg_2 = sum(1 for key in init_graph.edges(val) if key[0] not in val or key[1] not in val)
            if alg_1 < alg_2:
                model += pl.lpSum([
                    x[array_key[tuple(sorted(key))]]
                    for key in init_graph.edges(val) if key[0] in val and key[1] in val
                ]) <= len(val) - 1
            else:
                model += pl.lpSum([
                    x[array_key[tuple(sorted(key))]]
                    for key in init_graph.edges(val) if key[0] not in val or key[1] not in val
                ]) >= 2


def main():
    """Check the solution on test data."""

    n = 100
    # rnd.seed(777)
    points: List[Tuple[float, float]] = [(rnd.uniform(0, 1000), rnd.uniform(0, 1000)) for i in range(n)]

    start_time = datetime.now()
    # Build Delaunay triangulation for the list of points
    tri = Delaunay(points)
    base_graph = nx.Graph()
    for p in tri.simplices:
        base_graph.add_edge(p[0], p[1], weight=distance(points[p[0]], points[p[1]]))
        base_graph.add_edge(p[1], p[2], weight=distance(points[p[1]], points[p[2]]))
        base_graph.add_edge(p[0], p[2], weight=distance(points[p[0]], points[p[2]]))

    graph_united = nx.Graph()
    # Get the triangulation as a set of triangles, and save the edges into the graph
    for i, j in list(base_graph.edges()):
        for k in list(base_graph.adj[i]):
            if j != k and not graph_united.has_edge(j, k):
                graph_united.add_edge(j, k, weight=distance(points[j], points[k]))
        for k in list(base_graph.adj[j]):
            if i != k and not graph_united.has_edge(i, k):
                graph_united.add_edge(i, k, weight=distance(points[i], points[k]))

    print('Graph size:', n)
    print('Solver vars:', len(list(graph_united.edges())))

    # Return the solution graph
    graph_result = tsp_symmetric_linprog(graph_united)
    date_diff = (datetime.now() - start_time).total_seconds()

    if len(graph_result.edges()) > 0:
        # Calculate the exact length of the optimal path
        min_sum = sum([graph_united[i[0]][i[1]]['weight'] for i in graph_result.edges()])
        # Get the list of edges of the optimal path
        min_path = list(nx.find_cycle(graph_result, source=0))
        print('Path =', min_path)
    else:
        print('Solution impossible!')
    print('Solution time:', date_diff)

    # Draw the graph
    plt.figure(figsize=(9, 6), edgecolor='black', linewidth=1)
    plt.axis("equal")
    plt.title(f'Size: {n}; Length: {min_sum}', fontsize=10)
    # nx.draw(graph_result, points, width=0, with_labels=True, node_size=0, font_size=10, font_color="black")
    nx.draw(graph_result, points, width=4, edge_color="green", style="-", with_labels=False, node_size=0)
    # nx.draw(graph_united, points, width=0.5, edge_color="red", style="-", with_labels=False, node_size=0)
    nx.draw(base_graph, points, width=0.7, edge_color="blue", style="-", with_labels=False, node_size=0)
    plt.show()


if __name__ == "__main__":
    main()