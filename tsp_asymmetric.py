import pulp as pl
import math as math
import random as rnd
import networkx as nx
import matplotlib.pyplot as plt
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


def tsp_asymmetric_linprog(init_graph: nx.DiGraph) -> nx.DiGraph:
    """
    Solves the asymmetric Traveling Salesman Problem using integer linear programming.

    Parameters:
        init_graph (nx.DiGraph): The directed graph representing the distances.

    Returns:
        nx.DiGraph: The optimized route as a directed graph.
    """
    # Fill index dictionaries
    array_index: Dict[int, Tuple[int, int]] = {}
    array_key: Dict[Tuple[int, int], int] = {}
    for i, key in enumerate(init_graph.in_edges()):
        array_key[key] = i
        array_index[i] = key

    # Declare the model
    model = pl.LpProblem(name="tsp", sense=pl.LpMinimize)

    # Connect the solver
    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=86400)

    # Declare model variables
    x = [pl.LpVariable(name=f'x_{array_index[i][0]}_{array_index[i][1]}', cat='Binary') for i in array_index]

    # Input the objective function
    model += pl.lpSum([init_graph[array_index[i][0]][array_index[i][1]]['weight'] * x[i] for i in array_index])

    # Set initial constraints
    for i in range(len(init_graph)):
        # Each node can have only one outgoing edge
        model += pl.lpSum([x[array_key[j]] for j in init_graph.out_edges(i)]) == 1
        # Each node can have only one incoming edge
        model += pl.lpSum([x[array_key[j]] for j in init_graph.in_edges(i)]) == 1

    step = 0
    while True:
        # Find a solution
        status = model.solve(solver)
        step += 1

        if status != 1:
            return nx.DiGraph()

        # Save solver result as a graph
        graph_result = nx.DiGraph()
        for val in model.variables():
            if val.value() == 1:
                key = val.name.split('_')
                graph_result.add_edge(int(key[1]), int(key[2]))

        # Split the result graph into subsets
        result_sets = list(nx.connected_components(graph_result.to_undirected()))

        print(f'Step: {step}; Length: {round(model.objective.value())};')

        # Solution found if there is only one subset in the graph
        if len(result_sets) == 1:
            return graph_result

        # For each subset, add a constraint connecting it to other subsets
        for val in result_sets:
            # Choose the algorithm for creating a constraint that involves fewer variables
            alg_1 = sum(1 for key in init_graph.out_edges(val) if key[0] in val and key[1] in val)
            alg_2 = sum(1 for key in init_graph.out_edges(val) if key[1] not in val)
            if alg_1 < alg_2:
                model += pl.lpSum([x[array_key[key]] for key in init_graph.out_edges(val) if key[0] in val and key[1] in val]) <= len(val) - 1
            else:
                model += pl.lpSum([x[array_key[key]] for key in init_graph.out_edges(val) if key[1] not in val]) >= 1


def main():
    """Check the solution on test data."""

    n = 100
    # rnd.seed(777)

    points: List[Tuple[float, float]] = [(rnd.uniform(0, 1000), rnd.uniform(0, 1000)) for i in range(n)]

    # Take a symmetric problem as a basis and sparsify its edges
    init_graph = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if j > i:
                l = round(distance(points[i], points[j]))
                if rnd.randint(0, 100) < 50:
                    init_graph.add_edge(i, j, weight=l)
                if rnd.randint(0, 100) < 50:
                    init_graph.add_edge(j, i, weight=l)

    print('Graph size:', n)
    print('Solver vars:', len(init_graph.edges()))

    start_time = datetime.now()
    # Find the solution
    graph_result = tsp_asymmetric_linprog(init_graph)
    date_diff = (datetime.now() - start_time).total_seconds()

    if len(graph_result.edges()) > 0:
        # Calculate the exact length of the optimal path
        min_sum = sum([init_graph[i[0]][i[1]]['weight'] for i in graph_result.edges()])
        # Get the list of edges of the optimal path
        min_path = list(nx.find_cycle(graph_result, source=0))
        print('Path =', min_path)
    else:
        print('Solution impossible!')
    print('Solution time =', date_diff)

    # Draw the graph
    plt.figure(figsize=(9, 6), edgecolor='black', linewidth=1)
    plt.axis("equal")
    plt.title(f'Size: {n}; Length: {min_sum}', fontsize=10)
    # nx.draw(graph_result, points, width=0, with_labels=True, node_size=0, font_size=10, font_color="black", arrowsize=0.1)
    nx.draw(graph_result, points, width=1, edge_color="red", style="-", with_labels=False, node_size=0, arrowsize=8)
    plt.show()


if __name__ == "__main__":
    main()