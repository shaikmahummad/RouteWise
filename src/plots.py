"""Matplotlib visualizations for the supply chain optimizer."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def _fixed_positions(graph: nx.Graph) -> dict[str, tuple[float, float]]:
    warehouses = sorted(node for node, data in graph.nodes(data=True) if data["kind"] == "warehouse")
    hubs = sorted(node for node, data in graph.nodes(data=True) if data["kind"] == "hub")
    customers = sorted(node for node, data in graph.nodes(data=True) if data["kind"] == "customer")

    def column(nodes: list[str], x_value: float) -> dict[str, tuple[float, float]]:
        if len(nodes) == 1:
            heights = [0.0]
        else:
            heights = np.linspace(1.0, -1.0, len(nodes)).tolist()
        return {node: (x_value, float(y_value)) for node, y_value in zip(nodes, heights)}

    positions = column(warehouses, 0.0)
    positions.update(column(hubs, 1.0))
    positions.update(column(customers, 2.0))
    return positions


def _draw_nodes(
    axis: plt.Axes, graph: nx.Graph, positions: dict[str, tuple[float, float]], labels: dict[str, str]
) -> None:
    groups = [
        ("warehouse", "navy", "s"),
        ("hub", "grey", "o"),
        ("customer", "green", "o"),
    ]
    for kind, color, shape in groups:
        nodes = [node for node, data in graph.nodes(data=True) if data["kind"] == kind]
        nx.draw_networkx_nodes(
            graph, positions, nodelist=nodes, node_color=color, node_shape=shape,
            node_size=1050, ax=axis, edgecolors="white", linewidths=1.2,
        )
    nx.draw_networkx_labels(graph, positions, labels=labels, font_color="white", font_size=8, ax=axis)


def plot_network_topology(
    graph: nx.Graph, network_data: dict[str, Any]
) -> plt.Figure:
    """Plot the fixed-layout warehouse, hub, customer, and road network."""
    figure, axis = plt.subplots(figsize=(11, 6))
    positions = _fixed_positions(graph)
    nx.draw_networkx_edges(graph, positions, edge_color="#9aa0a6", width=1.5, ax=axis)
    edge_labels = {
        (left, right): f'{data["distance"]:g} km'
        for left, right, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_size=7, ax=axis)
    labels = {
        item["id"]: f'{item["id"]}\nSupply {item["supply"]}'
        for item in network_data["warehouses"]
    }
    labels.update({hub: hub for hub in network_data["hubs"]})
    labels.update(
        {item["id"]: f'{item["id"]}\nDemand {item["demand"]}' for item in network_data["customers"]}
    )
    _draw_nodes(axis, graph, positions, labels)
    axis.set_title("Supply Chain Road Network")
    axis.axis("off")
    figure.tight_layout()
    return figure


def plot_optimal_delivery_plan(
    graph: nx.Graph, road_loads: dict[tuple[str, str], float]
) -> plt.Figure:
    """Highlight used roads and scale their widths by transported units."""
    figure, axis = plt.subplots(figsize=(11, 6))
    positions = _fixed_positions(graph)
    used_edges = []
    unused_edges = []
    widths = []
    for left, right in graph.edges:
        load = road_loads.get(tuple(sorted((left, right))), 0.0)
        if load > 0:
            used_edges.append((left, right))
            widths.append(1.5 + 0.055 * load)
        else:
            unused_edges.append((left, right))
    nx.draw_networkx_edges(graph, positions, edgelist=unused_edges, edge_color="#b8bec5", width=1.2, ax=axis)
    if used_edges:
        nx.draw_networkx_edges(graph, positions, edgelist=used_edges, edge_color="red", width=widths, ax=axis)
    edge_labels = {(left, right): f'{data["distance"]:g} km' for left, right, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_size=7, ax=axis)
    _draw_nodes(axis, graph, positions, {node: node for node in graph.nodes})
    axis.set_title("Optimal Delivery Routes (red width reflects road load)")
    axis.axis("off")
    figure.tight_layout()
    return figure


def plot_demand_scenarios(scenarios: pd.DataFrame) -> plt.Figure:
    """Plot baseline and optimal costs for each demand scenario."""
    figure, axis = plt.subplots(figsize=(9, 5))
    feasible = scenarios[scenarios["Feasible"]]
    positions = np.arange(len(feasible))
    width = 0.36
    baseline = feasible["Baseline cost (Rs.)"].astype(float).to_numpy()
    optimal = feasible["Optimal cost (Rs.)"].astype(float).to_numpy()
    bars_baseline = axis.bar(positions - width / 2, baseline, width, label="Baseline")
    bars_optimal = axis.bar(positions + width / 2, optimal, width, label="Optimal")
    axis.set_xticks(positions, [f'{level}%' for level in feasible["Demand level (%)"]])
    axis.set_xlabel("Demand level")
    axis.set_ylabel("Transportation cost (Rs.)")
    axis.set_title("Demand Scenario Cost Comparison")
    axis.legend()
    axis.bar_label(bars_baseline, fmt="%.0f", padding=3, fontsize=8)
    axis.bar_label(bars_optimal, fmt="%.0f", padding=3, fontsize=8)
    figure.tight_layout()
    return figure


def plot_allocation_heatmap(
    allocation: np.ndarray, warehouses: list[str], customers: list[str]
) -> plt.Figure:
    """Display shipment quantities in a labelled allocation heatmap."""
    figure, axis = plt.subplots(figsize=(8, 4))
    image = axis.imshow(allocation, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(customers)), customers)
    axis.set_yticks(range(len(warehouses)), warehouses)
    axis.set_xlabel("Customers")
    axis.set_ylabel("Warehouses")
    axis.set_title("Optimal Allocation (units)")
    figure.colorbar(image, ax=axis, label="Units allocated")
    for row in range(allocation.shape[0]):
        for column in range(allocation.shape[1]):
            axis.text(column, row, f"{allocation[row, column]:g}", ha="center", va="center", color="black")
    figure.tight_layout()
    return figure
