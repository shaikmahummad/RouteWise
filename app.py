"""Interactive Streamlit dashboard for the supply chain optimizer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.module1_graph import build_graph, load_network_data
from src.module3_integration import analyze_network, demand_scenarios, road_criticality
from src.plots import (
    plot_allocation_heatmap,
    plot_demand_scenarios,
    plot_network_topology,
    plot_optimal_delivery_plan,
)


DATA_PATH = Path(__file__).parent / "data" / "network.json"
network_data = load_network_data(DATA_PATH)


def reset_controls() -> None:
    """Restore sidebar values to the original network inputs."""
    st.session_state["cost_per_km"] = float(network_data["cost_per_km"])
    st.session_state["demand_level"] = 100
    for warehouse in network_data["warehouses"]:
        st.session_state[f'supply_{warehouse["id"]}'] = int(warehouse["supply"])
    st.session_state["closed_road"] = "No road closure"


st.set_page_config(page_title="Supply Chain Route Optimizer", layout="wide")
st.title("Supply Chain & Delivery Route Optimizer")
st.caption("Graph theory, shortest paths, and linear programming for explainable delivery planning.")

with st.sidebar:
    st.header("Scenario controls")
    st.button("Reset to original data", on_click=reset_controls, use_container_width=True)
    cost_per_km = st.number_input(
        "Transportation cost (Rs. per unit-km)",
        min_value=0.0,
        value=float(network_data["cost_per_km"]),
        step=0.5,
        key="cost_per_km",
    )
    demand_level = st.slider(
        "Demand level (%)", min_value=50, max_value=120, value=100, step=1,
        key="demand_level",
    )
    supplies = []
    for warehouse in network_data["warehouses"]:
        warehouse_id = warehouse["id"]
        supplies.append(
            st.number_input(
                f"{warehouse_id} stock (units)",
                min_value=0,
                value=int(warehouse["supply"]),
                step=1,
                key=f"supply_{warehouse_id}",
            )
        )
    road_options = ["No road closure"] + [
        f'{road["from"]} - {road["to"]}' for road in network_data["roads"]
    ]
    selected_road = st.selectbox("Close one road", road_options, key="closed_road")

original_demands = [item["demand"] for item in network_data["customers"]]
selected_demands = [demand * demand_level / 100 for demand in original_demands]
closed_road = None
if selected_road != "No road closure":
    closed_road = tuple(selected_road.split(" - "))

analysis = analyze_network(
    network_data,
    cost_per_km=cost_per_km,
    supplies=supplies,
    demands=selected_demands,
    closed_road=closed_road,
)
tabs = st.tabs(["Network & Routes", "Optimal Plan", "What-if / Road Criticality", "Analysis"])

with tabs[0]:
    st.subheader("Original road network")
    st.pyplot(plot_network_topology(build_graph(network_data), network_data), clear_figure=True)
    column_a, column_b = st.columns(2)
    with column_a:
        st.markdown("**Warehouse supply (current controls)**")
        st.dataframe(pd.DataFrame({"Warehouse": analysis["warehouses"], "Supply": supplies}), hide_index=True)
    with column_b:
        st.markdown(f"**Customer demand ({demand_level}% of original)**")
        st.dataframe(
            pd.DataFrame({"Customer": analysis["customers"], "Demand": selected_demands}),
            hide_index=True,
        )
    if closed_road:
        st.info(f"Active calculations exclude road {selected_road}; the topology above remains original.")
    st.markdown("**Shortest distances (km; active network)**")
    st.dataframe(
        pd.DataFrame(
            analysis["distance_matrix"],
            index=analysis["warehouses"],
            columns=analysis["customers"],
        ).round(2)
    )
    st.markdown("**Transportation costs (Rs. per unit; active network)**")
    st.dataframe(
        pd.DataFrame(
            analysis["cost_matrix"],
            index=analysis["warehouses"],
            columns=analysis["customers"],
        ).round(2)
    )

with tabs[1]:
    optimal = analysis["optimal"]
    baseline = analysis["baseline"]
    if not optimal["success"]:
        st.error(f"Optimal plan is infeasible: {optimal['message']}")
    elif not baseline["success"]:
        st.error(f"Baseline plan is infeasible: {baseline['message']}")
    else:
        metric_columns = st.columns(3)
        metric_columns[0].metric("Optimal transportation cost", f"Rs. {optimal['total_cost']:,.2f}")
        metric_columns[1].metric("Nearest-warehouse baseline", f"Rs. {baseline['total_cost']:,.2f}")
        metric_columns[2].metric("Savings vs baseline", f'{analysis["savings_percent"]:.2f}%')
        st.pyplot(
            plot_optimal_delivery_plan(analysis["graph"], analysis["road_loads"]),
            clear_figure=True,
        )
        st.pyplot(
            plot_allocation_heatmap(
                optimal["allocation"], analysis["warehouses"], analysis["customers"]
            ),
            clear_figure=True,
        )
        st.markdown("**Positive-shipment route plan**")
        st.dataframe(analysis["route_plan"], hide_index=True)
        st.download_button(
            "Download route plan as CSV",
            data=analysis["route_plan"].to_csv(index=False).encode("utf-8"),
            file_name="optimal_route_plan.csv",
            mime="text/csv",
        )

with tabs[2]:
    if not analysis["optimal"]["success"]:
        st.error(f"Current scenario is infeasible; criticality cannot be measured: {analysis['optimal']['message']}")
    else:
        criticality = road_criticality(
            network_data,
            cost_per_km=cost_per_km,
            supplies=supplies,
            demands=selected_demands,
            closed_road=closed_road,
        )
        st.markdown(
            "Each road is closed independently and the shortest paths and LP are recalculated. "
            "A positive increase means the closure forces more expensive routes; infeasible closures "
            "are marked separately."
        )
        st.dataframe(criticality, hide_index=True)
        feasible_criticality = criticality[criticality["Feasible"]].head(3)
        if not feasible_criticality.empty:
            st.markdown("**Three most critical feasible roads**")
            st.dataframe(feasible_criticality, hide_index=True)
        infeasible_count = int((~criticality["Feasible"]).sum())
        if infeasible_count:
            st.warning(f"{infeasible_count} road closure(s) make the current scenario infeasible.")

with tabs[3]:
    scenarios = demand_scenarios(
        network_data,
        cost_per_km=cost_per_km,
        supplies=supplies,
        closed_road=closed_road,
    )
    st.pyplot(plot_demand_scenarios(scenarios), clear_figure=True)
    st.dataframe(scenarios, hide_index=True)
    if analysis["optimal"]["success"]:
        st.subheader("Constraint shadow prices")
        shadow_columns = st.columns(2)
        shadow_columns[0].dataframe(
            pd.DataFrame(
                {
                    "Warehouse": analysis["warehouses"],
                    "Supply shadow price (Rs./unit)": analysis["optimal"]["supply_shadow_prices"],
                }
            ),
            hide_index=True,
        )
        shadow_columns[1].dataframe(
            pd.DataFrame(
                {
                    "Customer": analysis["customers"],
                    "Demand shadow price (Rs./unit)": analysis["optimal"]["demand_shadow_prices"],
                }
            ),
            hide_index=True,
        )
        st.caption(
            "A positive supply shadow price estimates the cost saving from one extra supply unit "
            "within the current optimal basis. Demand prices estimate the marginal cost of one "
            "additional demanded unit."
        )
    else:
        st.info("Shadow prices are unavailable because the current optimization is infeasible.")
    with st.expander("Mathematical LP formulation"):
        st.latex(r"\min \; Z = \sum_{i=1}^{m}\sum_{j=1}^{n} c_{ij}x_{ij}")
        st.latex(r"\sum_{j=1}^{n}x_{ij}\leq s_i \quad \forall i")
        st.latex(r"\sum_{i=1}^{m}x_{ij}=d_j \quad \forall j")
        st.latex(r"x_{ij}\geq 0 \quad \forall i,j")

