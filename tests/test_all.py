"""Focused tests for graph, optimization, and integration behavior."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.module1_graph import build_graph, dijkstra, load_network_data, reconstruct_path
from src.module3_integration import analyze_network, demand_scenarios, road_criticality


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def network():
    return load_network_data(ROOT / "data" / "network.json")


def test_dijkstra_matches_networkx_for_all_nodes(network):
    graph = build_graph(network)
    for start in graph.nodes:
        actual, _ = dijkstra(graph, start)
        expected = nx.single_source_dijkstra_path_length(graph, start, weight="distance")
        for node in graph.nodes:
            assert actual[node] == pytest.approx(expected.get(node, float("inf")))
    distances, predecessors = dijkstra(graph, "W1")
    route = reconstruct_path(predecessors, "W1", "C5")
    assert route[0] == "W1"
    assert route[-1] == "C5"
    assert sum(graph[left][right]["distance"] for left, right in zip(route, route[1:])) == pytest.approx(distances["C5"])


def test_lp_supply_demand_and_nonnegativity(network):
    result = analyze_network(network)
    assert result["optimal"]["success"]
    allocation = result["optimal"]["allocation"]
    assert np.all(allocation >= -1e-9)
    assert np.all(allocation.sum(axis=1) <= np.array(result["supplies"]) + 1e-8)
    assert allocation.sum(axis=0) == pytest.approx(result["demands"])


def test_expected_baseline_and_optimal_results(network):
    result = analyze_network(network, cost_per_km=2)
    assert result["optimal"]["total_cost"] == pytest.approx(31350)
    assert result["baseline"]["total_cost"] == pytest.approx(32700)
    assert result["savings_percent"] == pytest.approx(4.12844036697)


def test_110_percent_demand_is_infeasible(network):
    demands = [item["demand"] * 1.1 for item in network["customers"]]
    result = analyze_network(network, demands=demands)
    assert not result["optimal"]["success"]
    assert "exceeds total supply" in result["optimal"]["message"]


def test_closing_h3_c3_has_expected_optimal_cost(network):
    result = analyze_network(network, closed_road=("H3", "C3"))
    assert result["optimal"]["success"]
    assert result["optimal"]["total_cost"] == pytest.approx(31650)


def test_supply_shadow_prices(network):
    result = analyze_network(network)
    prices = result["optimal"]["supply_shadow_prices"]
    assert prices[1] == pytest.approx(80)
    assert prices[2] == pytest.approx(130)


def test_selected_road_closure_flows_into_scenario_analyses(network):
    scenarios = demand_scenarios(network, levels=(100,), closed_road=("H3", "C3"))
    assert scenarios.iloc[0]["Optimal cost (Rs.)"] == pytest.approx(31650)

    criticality = road_criticality(network, closed_road=("H3", "C3"))
    selected_road = criticality[criticality["Road"] == "H3 - C3"].iloc[0]
    assert selected_road["Cost increase (Rs.)"] == pytest.approx(0)
