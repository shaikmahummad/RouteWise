"""Graph construction and manually implemented shortest paths."""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


def load_network_data(path: str | Path) -> dict[str, Any]:
    """Load the network definition from a JSON file."""
    with Path(path).open(encoding="utf-8") as network_file:
        return json.load(network_file)


def build_graph(network_data: dict[str, Any]) -> nx.Graph:
    """Build an undirected road graph with distances in kilometres."""
    graph = nx.Graph()
    for warehouse in network_data["warehouses"]:
        graph.add_node(warehouse["id"], kind="warehouse", supply=warehouse["supply"])
    for hub in network_data["hubs"]:
        graph.add_node(hub, kind="hub")
    for customer in network_data["customers"]:
        graph.add_node(customer["id"], kind="customer", demand=customer["demand"])
    for road in network_data["roads"]:
        graph.add_edge(road["from"], road["to"], distance=float(road["distance"]))
    return graph


def dijkstra(
    graph: nx.Graph, start: str
) -> tuple[dict[str, float], dict[str, str | None]]:
    """Return shortest distances and predecessors using heapq-based Dijkstra."""
    distances = {node: float("inf") for node in graph.nodes}
    predecessors: dict[str, str | None] = {node: None for node in graph.nodes}
    if start not in graph:
        return distances, predecessors

    distances[start] = 0.0
    queue: list[tuple[float, str]] = [(0.0, start)]
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, edge_data in graph[current_node].items():
            candidate = current_distance + float(edge_data["distance"])
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                predecessors[neighbor] = current_node
                heapq.heappush(queue, (candidate, neighbor))
    return distances, predecessors


def reconstruct_path(
    predecessors: dict[str, str | None], start: str, target: str
) -> list[str]:
    """Reconstruct a shortest path, or return an empty list if unreachable."""
    if start not in predecessors or target not in predecessors:
        return []
    route = [target]
    current = target
    while current != start:
        current_predecessor = predecessors.get(current)
        if current_predecessor is None:
            return []
        current = current_predecessor
        route.append(current)
    route.reverse()
    return route


def shortest_distance_matrices(
    graph: nx.Graph, warehouses: list[str], customers: list[str]
) -> tuple[np.ndarray, dict[tuple[str, str], list[str]]]:
    """Compute warehouse-customer distances and their shortest road paths."""
    matrix = np.full((len(warehouses), len(customers)), np.inf, dtype=float)
    paths: dict[tuple[str, str], list[str]] = {}
    for warehouse_index, warehouse in enumerate(warehouses):
        distances, predecessors = dijkstra(graph, warehouse)
        for customer_index, customer in enumerate(customers):
            matrix[warehouse_index, customer_index] = distances.get(customer, np.inf)
            paths[(warehouse, customer)] = reconstruct_path(
                predecessors, warehouse, customer
            )
    return matrix, paths


def transportation_cost_matrix(
    distance_matrix: np.ndarray, cost_per_km: float
) -> np.ndarray:
    """Convert route distances into per-unit transportation costs."""
    return distance_matrix * cost_per_km


def validate_dijkstra_with_networkx(graph: nx.Graph, start: str) -> bool:
    """Compare manual distances to NetworkX's implementation for validation."""
    distances, _ = dijkstra(graph, start)
    for node in graph.nodes:
        expected = (
            nx.dijkstra_path_length(graph, start, node, weight="distance")
            if nx.has_path(graph, start, node)
            else float("inf")
        )
        if not np.isclose(distances[node], expected):
            return False
    return True
