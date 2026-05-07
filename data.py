import json

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon


def generate_grid_graph(N, filename, population_matrix=None, *, num_districts: int):
    nodes = []
    adjacency = [[] for _ in range(N * N)]
    pop = None
    if population_matrix is not None:
        pop = np.asarray(population_matrix)

    def node_id(x, y):
        return x * N + y

    for x in range(N):
        for y in range(N):
            idx = node_id(x, y)
            border_length = 0
            if x == 0:
                border_length += 1
            if x == N - 1:
                border_length += 1
            if y == 0:
                border_length += 1
            if y == N - 1:
                border_length += 1

            node = {
                "id": idx,
                "precinct_id": idx,
                "precinct_id_str": f"({x},{y})",
                "border_length": border_length,
                "x_location": x,
                "y_location": y,
                "area": 1,
                "population": int(pop[x, y]) if pop is not None else 1,
                "county": "A",
            }
            nodes.append(node)
            if x > 0:
                adjacency[idx].append({"id": node_id(x - 1, y), "length": 1})
            if x < N - 1:
                adjacency[idx].append({"id": node_id(x + 1, y), "length": 1})
            if y > 0:
                adjacency[idx].append({"id": node_id(x, y - 1), "length": 1})
            if y < N - 1:
                adjacency[idx].append({"id": node_id(x, y + 1), "length": 1})

    graph_json = {
        "directed": False,
        "multigraph": False,
        "graph": [],
        "num_districts": int(num_districts),
        "nodes": nodes,
        "adjacency": adjacency,
    }
    with open(filename, "w") as f:
        json.dump(graph_json, f, indent=4)
    return graph_json


def generate_grid_shape(N, cell_size=1.0, *, crs="EPSG:3857"):
    polys = []
    ids = []
    precincts = []
    for i in range(N):
        for j in range(N):
            x0, y0 = j * cell_size, i * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            polys.append(poly)
            ids.append(i * N + j)
            precincts.append(f"({i},{j})")
    return gpd.GeoDataFrame({"id": ids, "precinct": precincts}, geometry=polys, crs=crs)
