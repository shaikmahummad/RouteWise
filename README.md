# Supply Chain & Delivery Route Optimizer Using Graph Theory and Linear Programming

## Problem statement

This capstone models a small supply network of warehouses, distribution hubs, customers, and undirected roads. It finds economical shipment quantities while routing deliveries over the shortest available road paths.

## Project objectives

- Represent the road network as a weighted graph.
- Calculate warehouse-to-customer shortest distances with manually implemented Dijkstra.
- Minimize transportation cost using a linear program.
- Compare the optimum with a nearest-warehouse heuristic.
- Study demand scenarios and individual road closures in an interactive dashboard.

## Technologies used

Python 3.11+, NumPy, SciPy (`linprog` with HiGHS), NetworkX, Matplotlib, Pandas, Streamlit, and Pytest.

## Project folder structure

```text
supply_chain_optimizer/
├── data/
│   └── network.json
├── src/
│   ├── __init__.py
│   ├── module1_graph.py
│   ├── module2_lp.py
│   ├── module3_integration.py
│   └── plots.py
├── tests/
│   └── test_all.py
├── app.py
├── requirements.txt
└── README.md
```

## How It Works

1. The application loads warehouse, hub, customer, and road data from JSON.
2. Dijkstra's algorithm finds each warehouse-to-customer shortest route.
3. Route distances become per-unit transportation costs.
4. Linear programming chooses shipment quantities that meet demand at minimum cost.
5. The dashboard compares the plan with a heuristic and explores changes in demand and roads.

## Module descriptions

- `module1_graph.py` loads the JSON data, builds the undirected weighted graph, implements Dijkstra with `heapq`, reconstructs paths, and generates distance and cost matrices.
- `module2_lp.py` constructs supply and demand constraints and solves the transportation model with SciPy's HiGHS solver. Unreachable routes are prohibited by zero-capacity variable bounds.
- `module3_integration.py` combines graph distances and optimization, calculates a nearest-warehouse baseline, expands shipments into road loads, and evaluates demand scenarios and road criticality.
- `plots.py` provides four independent Matplotlib figures for the network, delivery plan, scenarios, and allocation matrix.

## Graph theory explanation

Each warehouse, hub, and customer is a vertex. Each road is an undirected edge weighted by its distance in kilometres. The graph captures alternate routes, and removing an edge models a road closure. NetworkX stores the graph and validates results; the actual shortest-path calculation is implemented separately.

## Dijkstra algorithm explanation

Dijkstra's algorithm starts a selected warehouse at distance zero and all other vertices at infinity. A min-priority queue repeatedly selects the closest unsettled vertex and relaxes its adjacent edges. The resulting distance and predecessor mappings produce shortest distances and reconstructable paths. Non-negative road lengths make Dijkstra's method applicable.

## Linear programming mathematical model

Let `x_ij` be the units shipped from warehouse `i` to customer `j`, `s_i` the available stock, `d_j` the required demand, and `c_ij` the cost per unit over the shortest route.

```text
Minimize    Z = sum_i sum_j c_ij x_ij
Subject to  sum_j x_ij <= s_i       for each warehouse i
            sum_i x_ij = d_j        for each customer j
            x_ij >= 0               for every i,j
```

The supply limits allow unused inventory; all customer demand must be fulfilled. An unreachable warehouse-customer pair is assigned a fixed zero shipment bound. Supply shadow prices are reported as the negative of HiGHS' inequality marginals, so a positive value represents estimated cost savings from one additional supply unit. Demand shadow prices use the solver's equality marginals and represent marginal cost changes.

## Baseline heuristic

Customers are processed from largest demand to smallest. For each, the heuristic allocates as much as possible from the nearest warehouse that still has stock, then proceeds to the next nearest warehouse if necessary. The resulting feasible allocation is costed using the same shortest-path costs as the optimum. The heuristic is a comparison, not a guarantee of global optimality.

## Road criticality

The analysis closes one road at a time, recomputes shortest routes and transportation costs, and solves the LP again. Cost increase is the closed-road optimum minus the original optimum. Roads are ranked by this increase; closures that prevent demand from being served are reported as infeasible instead of given a fabricated cost.

## Key results

With the supplied data and Rs. 2 per unit-kilometre, the expected optimum is Rs. 31,350 and the nearest-warehouse baseline is Rs. 32,700, corresponding to approximately 4.1% savings. Closing H3-C3 is expected to increase the optimum to Rs. 31,650. At 110% demand, total demand exceeds available supply, so the LP is infeasible. These values are calculated from the data at runtime, not hardcoded.

## Installation

From the project root, use Python 3.11 or newer:

```powershell
python -m pip install -r requirements.txt
```

## Execution commands

```powershell
python -m streamlit run app.py
```

## Testing

```powershell
python -m pytest tests/test_all.py -v
```

Tests compare manual Dijkstra distances with NetworkX, check LP feasibility and expected costs, verify the road closure, and validate supply shadow prices.

## Expected dashboard functionality

The four tabs show the network and matrices, the optimized plan and downloadable route table, road-closure criticality, and demand-scenario/shadow-price analysis. Sidebar controls adjust the transportation rate, demand level, each warehouse's stock, and one road closure. Reset returns the controls to the supplied values without editing the JSON file.
