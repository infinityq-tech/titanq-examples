# README

#### This project gives examples for solving the Minimum Directed Cycle Problem under the context of Currency Arbitrage using the TitanQ SDK.
--------------------------------------------------------------------------------


Here is an overview:
- Optimal Arbitrage Opportunities

- Problem Formulation

- Hyperparameter Tuning

- TitanQ Example

- License

## Optimal Arbitrage Opportunities

Given a list of financial assets/currencies and some exchange rates between them, one may want to find some series of exchanges between currencies that will generate profit.

We can model the financial market as a directed graph, where nodes represent currencies or assets, and directed edges represent exchange rates. The goal is to identify a series of exchanges to gain profit by identifying an appropriate cycle in the graph. An arbitrage opportunity exists if the product of the exchange rates in the cycle is greater than 1 (or equivalently, the sum of negative log exchange rates is negative).

In our examples, we focus on *currencies*, with an option to load a full dataset from a given day January 2, 2025. As a simple example, consider the set of 4 currencies: USD, EUR, JPY, and GBP. Below we find the exchange rates between each pair of currencies from current data. Each entry i,j represents the exchange rate from currency i to currency j. For example, we have that the exchange rate from EUR to USD is 1.04297, meaning 1 EUR = 1.04297 USD. Edges that do not exist are assigned an exchange rate of zero, such as for USD/USD or missing data. This is to prevent an invalid currency exchange.
```
~~~~~~~~~~~~~ Exchange Rate Matrix ~~~~~~~~~~~~~
          USD       EUR         JPY       GBP  
USD       0.0    0.9588  155.192001   0.80362
EUR   1.04297       0.0  161.785004    0.8377
JPY  0.006444  0.006181         0.0  0.005173
GBP  1.244369   1.19353  193.117004       0.0
```
Due to the directed nature of the graph, there can be two edges between a pair of currencies in different directions. For this small example, every distinct pair has two edges between them, but this may not always be the case.

Recall our question from earlier: *Given a set of assets in which we could potentially exchange between, how do we identify the most profitable arbitrage opportunity?* Using the example above, this translates to finding a directed cycle where the product of all exchanges is maximized and exceeds 1.

Imagine your cycle is ['USD/EUR', 'EUR/JPY', 'JPY/USD']. Then your exchanges are $0.9588 \times 161.785004 \times 0.006444 = 0.9995898120660287$. 

Hence there is a net loss since the result is 0.9995898120660287x your initial value. However, there would be a profit if the product had been greater than 1.

## Problem Formulation

We can simplify our problem to only consider edges that exist in our graph. Thus we take the indices of all existing edges and store them in a list, and a corresponding list of edge names to help keep track of which edges we are using.

For example, we have the list of non-zero exchange rates to be 
```
[0.9588, 155.192001, 0.80362, 1.04297,  161.785004,  0.8377,  0.006444,  0.006181,  0.005173,  1.244369,  1.19353,  193.117004]
```
and the list of corresponding ```edges``` to be 
```
edges = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)]
edge_names = ['USD/EUR', 'USD/JPY', 'USD/GBP', 'EUR/USD', 'EUR/JPY', 'EUR/GBP', 'JPY/USD', 'JPY/EUR', 'JPY/GBP', 'GBP/USD', 'GBP/EUR', 'GBP/JPY'].
```

We can solve the problem in TitanQ by setting our decision variable ```x```. Since the variable in question is deciding which edges to select, we can naturally model this as a vector of binary variables. We define
```math
x = \begin{bmatrix} x_{ij} \end{bmatrix} ~ \text{ for all (i, j)} \in \{edges\}
```
where $x_{ij} = 0$ indicates that edge $ij$ *is not* selected for the cycle, and $x_{ij} = 1$ indicates that edge $ij$ *is* selected for the cycle. Note that x is the size of ```edges```.

To calculate profit in an exchange cycle, we start with a unit of 1 and multiply by all the exchange rates in the cycle. If we take the log of this product, it can be equivalently rewritten as the summation of the log of each exchange rate. To convert this into a minimization problem, we also take the negative of each of those values.

Thus, we define the *bias vector* in TitanQ to simply be a vector of the length of ```x``` containing the negative log weights of the exchange values.

```
~~~~~~~~~~~~~ Bias Vector ~~~~~~~~~~~~~
b = [0.042073, -5.044663, 0.218629, -0.042073,  -5.086268,  0.177095,  5.044663,  5.086340,  5.264302,  -0.218629,  -0.176915,  -5.263296]
```
where each $b_{ij}$ is the negative log of the exchange rate from currency $i$ to currency $j$, for all (i, j) in edges.

Then the objective function to minimize is simply:

```math
b^Tx
```

To ensure our solution ```x``` represents a cycle, we first implement a constraint that restricts the total number of purchases of a currency to be equal to the number of sales of that same currency. We also forbid exchanging through a currency more than once.

This can be written as the following two constraints:

(1) $\sum_{j, (i, j) \in E} x_{ij} - \sum_{j, (j, i) \in E} x_{ji} = 0  ~ \text{ for all } i \in V$

(2) $\sum_{j, (i, j) \in E} x_{ij} \leq 1  ~ \text{ for all } i \in V$

We call constraint (1) our flow constraint and constraint (2) our number of exchanges constraint.

## Hyperparameter Tuning

The hyperparameters used to tune the TitanQ solver are the following:

- *beta* = Scales the problem by this factor (inverse of temperature). A lower *beta* allows for easier escape from local minima, while a higher *beta* is more likely to respect penalties and constraints.

- *timeout_in_secs* = Maximum runtime of the solver in seconds.

- *num_chains* = Number of parallel chains running computations within each engine.

- *num_engines* = Number of independent parallel problems to run.

- *penalty_scaling* = Scales the impact of constraint violations. A higher *penalty_scaling* means a higher likelihood of generating a feasible solution.

## TitanQ Example

*instances*: There are two instances provided with a list of currencies (*currencies_6*, *currencies_25*). A full dataset for a given day January 2, 2025 can also be loaded if desired.

*exch_rates*: Exchange rates for currencies_6 and currencies_25.

*arbitrage_titanq.ipynb*: This Jupyter notebook demonstrates a full step-by-step use of the TitanQ SDK, either from *currencies_6* or *currencies_25* or a full dataset for January 2, 2025.

*model_generation.py*: Functions used to create, solve, and analyze results from the TitanQ model.

The required packages are listed in *requirements.txt* and can be installed using pip:

```bash
pip install -r requirements.txt
```

## License

Released under the Apache License 2.0. See [LICENSE](../LICENSE) file.
