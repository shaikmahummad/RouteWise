"""Integration of route calculations, optimization, and scenario analysis."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.module1_graph import (
    build_graph,
    shortest_distance_matrices,
    transportation_cost_matrix,
)
from src.module2_lp import solve_transportation

ROUTE_COLUMNS = [
    "From",
    "To",
    "Units",
    "Road path",
    "Distance (km)",
    "Transportation cost (Rs.)",
]


def _network_ids(network_data: dict[str, Any]) -> tuple[list[str], list[str]]:
    return (
        [item["id"] for item in network_data["warehouses"]],
        [item["id"] for item in network_data["customers"]],
    )


def _baseline(
    distance_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    warehouses: list[str],
    customers: list[str],
    supplies: list[float],
    demands: list[float],
) -> dict[str, object]:
    """Allocate each largest-demand customer from its nearest available stock."""
    remaining = np.asarray(supplies, dtype=float).copy()
    allocation = np.zeros((len(warehouses), len(customers)), dtype=float)
    for customer_index in sorted(
        range(len(customers)), key=lambda index: (-demands[index], index)
    ):
        outstanding = float(demands[customer_index])
        warehouse_order = sorted(
            range(len(warehouses)),
            key=lambda index: (distance_matrix[index, customer_index], index),
        )
        for warehouse_index in warehouse_order:
            if not np.isfinite(distance_matrix[warehouse_index, customer_index]):
                continue
            shipped = min(remaining[warehouse_index], outstanding)
            allocation[warehouse_index, customer_index] += shipped
            remaining[warehouse_index] -= shipped
            outstanding -= shipped
            if outstanding <= 1e-9:
                break
        if outstanding > 1e-9:
            return {
                "success": False,
                "message": f"Baseline cannot satisfy demand for {customers[customer_index]}.",
                "allocation": None,
                "total_cost": None,
            }
    return {
        "success": True,
        "message": "All customer demands were satisfied.",
        "allocation": allocation,
        "total_cost": float(np.sum(allocation * np.where(np.isfinite(cost_matrix), cost_matrix, 0))),
    }


def analyze_network(
    network_data: dict[str, Any],
    cost_per_km: float | None = None,
    supplies: list[float] | None = None,
    demands: list[float] | None = None,
    closed_road: tuple[str, str] | None = None,
    closed_roads: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Calculate routes, optimal and baseline plans, loads, and comparison."""
    warehouses, customers = _network_ids(network_data)
    supply_values = supplies if supplies is not None else [
        item["supply"] for item in network_data["warehouses"]
    ]
    demand_values = demands if demands is not None else [
        item["demand"] for item in network_data["customers"]
    ]
    rate = float(network_data["cost_per_km"] if cost_per_km is None else cost_per_km)
    graph = build_graph(network_data)
    roads_to_close = list(closed_roads or [])
    if closed_road is not None:
        roads_to_close.append(closed_road)
    for road in roads_to_close:
        if graph.has_edge(*road):
            graph.remove_edge(*road)

    distances, paths = shortest_distance_matrices(graph, warehouses, customers)
    costs = transportation_cost_matrix(distances, rate)
    optimal = solve_transportation(costs, supply_values, demand_values)
    baseline = _baseline(
        distances, costs, warehouses, customers, supply_values, demand_values
    )
    result: dict[str, Any] = {
        "graph": graph,
        "warehouses": warehouses,
        "customers": customers,
        "supplies": list(supply_values),
        "demands": list(demand_values),
        "cost_per_km": rate,
        "distance_matrix": distances,
        "cost_matrix": costs,
        "paths": paths,
        "optimal": optimal,
        "baseline": baseline,
        "road_loads": {},
        "route_plan": pd.DataFrame(columns=ROUTE_COLUMNS),
        "absolute_cost_difference": None,
        "savings_percent": None,
    }

    if not optimal["success"]:
        return result
    allocation = optimal["allocation"]
    road_loads: dict[tuple[str, str], float] = {}
    route_rows: list[dict[str, object]] = []
    for warehouse_index, warehouse in enumerate(warehouses):
        for customer_index, customer in enumerate(customers):
            units = float(allocation[warehouse_index, customer_index])
            if units <= 1e-9:
                continue
            path = paths[(warehouse, customer)]
            distance = float(distances[warehouse_index, customer_index])
            for left, right in zip(path, path[1:]):
                edge_key = tuple(sorted((left, right)))
                road_loads[edge_key] = road_loads.get(edge_key, 0.0) + units
            route_rows.append(
                {
                    "From": warehouse,
                    "To": customer,
                    "Units": units,
                    "Road path": " -> ".join(path),
                    "Distance (km)": distance,
                    "Transportation cost (Rs.)": units * costs[warehouse_index, customer_index],
                }
            )
    result["road_loads"] = road_loads
    result["route_plan"] = pd.DataFrame(route_rows, columns=ROUTE_COLUMNS)
    if baseline["success"]:
        baseline_cost = float(baseline["total_cost"])
        optimal_cost = float(optimal["total_cost"])
        result["absolute_cost_difference"] = abs(baseline_cost - optimal_cost)
        result["savings_percent"] = (
            (baseline_cost - optimal_cost) / baseline_cost * 100
            if baseline_cost
            else 0.0
        )
    return result


def demand_scenarios(
    network_data: dict[str, Any],
    cost_per_km: float | None = None,
    supplies: list[float] | None = None,
    levels: tuple[int, ...] = (80, 90, 100, 110),
    closed_road: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Compare baseline and optimal costs for demand percentages."""
    original_demands = [item["demand"] for item in network_data["customers"]]
    rows = []
    for level in levels:
        scaled_demands = [demand * level / 100 for demand in original_demands]
        result = analyze_network(
            network_data,
            cost_per_km=cost_per_km,
            supplies=supplies,
            demands=scaled_demands,
            closed_road=closed_road,
        )
        optimal = result["optimal"]
        baseline = result["baseline"]
        rows.append(
            {
                "Demand level (%)": level,
                "Total demand": sum(scaled_demands),
                "Optimal cost (Rs.)": optimal["total_cost"] if optimal["success"] else None,
                "Baseline cost (Rs.)": baseline["total_cost"] if baseline["success"] else None,
                "Feasible": bool(optimal["success"] and baseline["success"]),
                "Status": optimal["message"] if not optimal["success"] else baseline["message"],
            }
        )
    return pd.DataFrame(rows)


def road_criticality(
    network_data: dict[str, Any],
    cost_per_km: float | None = None,
    supplies: list[float] | None = None,
    demands: list[float] | None = None,
    closed_road: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Measure the optimal cost change after individually closing every road."""
    original = analyze_network(
        network_data, cost_per_km, supplies, demands, closed_road=closed_road
    )
    original_optimal = original["optimal"]
    already_closed = [closed_road] if closed_road is not None else []
    rows = []
    for road in network_data["roads"]:
        left, right = road["from"], road["to"]
        tested_closures = already_closed + [(left, right)]
        closed = analyze_network(
            network_data,
            cost_per_km,
            supplies,
            demands,
            closed_roads=tested_closures,
        )["optimal"]
        feasible = bool(original_optimal["success"] and closed["success"])
        closed_cost = closed["total_cost"] if closed["success"] else None
        rows.append(
            {
                "Road": f"{left} - {right}",
                "Distance (km)": road["distance"],
                "Closed-road optimal cost (Rs.)": closed_cost,
                "Cost increase (Rs.)": (
                    float(closed_cost) - float(original_optimal["total_cost"])
                    if feasible
                    else None
                ),
                "Feasible": feasible,
                "Status": closed["message"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        "Cost increase (Rs.)", ascending=False, na_position="last"
    ).reset_index(drop=True)
