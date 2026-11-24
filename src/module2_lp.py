"""Linear programming model for the transportation problem."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def solve_transportation(
    cost_matrix: np.ndarray,
    supplies: list[float],
    demands: list[float],
) -> dict[str, object]:
    """Minimize transportation cost subject to supply and demand constraints."""
    costs = np.asarray(cost_matrix, dtype=float)
    supply_values = np.asarray(supplies, dtype=float)
    demand_values = np.asarray(demands, dtype=float)
    warehouse_count, customer_count = costs.shape

    def infeasible(message: str) -> dict[str, object]:
        return {
            "success": False,
            "message": message,
            "allocation": None,
            "total_cost": None,
            "supply_shadow_prices": None,
            "demand_shadow_prices": None,
        }

    if costs.shape != (len(supplies), len(demands)):
        return infeasible("Cost matrix dimensions do not match supplies and demands.")
    if np.any(supply_values < 0) or np.any(demand_values < 0):
        return infeasible("Supply and demand values must be non-negative.")
    if np.sum(demand_values) > np.sum(supply_values) + 1e-9:
        return infeasible(
            f"Infeasible: total demand ({np.sum(demand_values):g}) exceeds "
            f"total supply ({np.sum(supply_values):g})."
        )
    if np.any(np.isnan(costs)) or np.any(costs < 0):
        return infeasible("Transportation costs must be non-negative and not NaN.")

    objective = np.where(np.isfinite(costs), costs, 0.0).reshape(-1)
    bounds = [
        (0, None) if np.isfinite(costs[i, j]) else (0, 0)
        for i in range(warehouse_count)
        for j in range(customer_count)
    ]
    supply_constraints = np.zeros((warehouse_count, warehouse_count * customer_count))
    for warehouse_index in range(warehouse_count):
        start = warehouse_index * customer_count
        supply_constraints[warehouse_index, start : start + customer_count] = 1
    demand_constraints = np.zeros((customer_count, warehouse_count * customer_count))
    for customer_index in range(customer_count):
        demand_constraints[
            customer_index, customer_index :: customer_count
        ] = 1

    result = linprog(
        c=objective,
        A_ub=supply_constraints,
        b_ub=supply_values,
        A_eq=demand_constraints,
        b_eq=demand_values,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return infeasible(f"Optimization failed: {result.message}")

    allocation = result.x.reshape(warehouse_count, customer_count)
    return {
        "success": True,
        "message": result.message,
        "allocation": allocation,
        "total_cost": float(result.fun),
        # HiGHS marginals are cost derivatives; negate them to report the
        # positive cost saving from one additional unit of supply.
        "supply_shadow_prices": -np.asarray(result.ineqlin.marginals, dtype=float),
        "demand_shadow_prices": np.asarray(result.eqlin.marginals, dtype=float),
    }
