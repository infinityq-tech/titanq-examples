# README

#### This project gives examples for solving the MWIS Problem under the context of Portfolio Optimization using the TitanQ SDK.
--------------------------------------------------------------------------------


Here is an overview:

- Portfolio Optimization

- Maximum Weighted Independent Set

- Problem Formulation Using QUBO 

- Hyperparameter Tuning

- TitanQ Example

- License

## Portfolio Optimization

When constructing a portfolio of assets (e.g. stocks) one of the main goals is to maximize expected return while minimizing risk (variance). One way to minimize risk is to invest in assets that have low correlation with each other (uncorrelated assets).

>***Given a set of assets in which we could potentially invest in, how do we identify a subset of uncorrelated assets?***

First, we can construct a *market correlation graph* by using the assets as nodes and connecting two nodes with an edge if they are *correlated* in some way. One direct way to determine if two assets are correlated is if the absolute value of the correlation coefficient of their returns is greater than (or equal to) a threshold $\theta$. That is, we have
```math
\text{asset } i \text{ and } \text{asset } j \text{ are connected } \Leftrightarrow |correlation(r_i, r_j)| \geq \theta
```
where $r_i$ and $r_j$ are the respective returns of the assets. Here $\theta$ is an investor chosen parameter such that $0 \leq \theta \leq 1$. The correlation coefficient can be calculated using historical data over an investor chosen time period and frequency (e.g. annual returns from 2010 start to 2023 end). Creating edges this way effectively connects assets that are either somewhat positively correlated or negatively correlated based on the threshold $\theta$. On the other hand, assets that are not connected by an edge are considered uncorrelated. Note that lower values of $\theta$ will lead to more edges in the graph (i.e. greater graph density) and vice versa.

Moreover, the nodes can be weighted based on the investor's preference for each asset, which could be a combination of historical returns, familiarity with the industry etc. A greater weight would correspond to a higher preference to invest in that asset.

In our examples, we focus on *stock* portfolios, and take advantage of the library *yfinance* to obtain historical data on stocks. As a simple example, consider the set of seven stocks: AAPL (apple), SHEL (Shell), TSLA (Tesla), GIS	(General Mills), MCD (McDonald's), KO (Coca-Cola), BB (BlackBerry). Below we calculate the correlation coefficients between the annual returns of each pair of stocks from January 2010 to December 2023.

```
~~~~~~~~~~~~~ Correlation Matrix ~~~~~~~~~~~~~
Ticker      AAPL      SHEL      TSLA       GIS       MCD        KO        BB
Ticker                                                                      
AAPL    0.000000 -0.220904  0.461838  0.175364  0.139766  0.088384  0.517539
SHEL   -0.220904  0.000000 -0.564393  0.113255  0.154930  0.209730  0.018272
TSLA    0.461838 -0.564393  0.000000  0.093117 -0.029303 -0.098812  0.119746
GIS     0.175364  0.113255  0.093117  0.000000  0.043111  0.610211 -0.165095
MCD     0.139766  0.154930 -0.029303  0.043111  0.000000  0.260313  0.252453
KO      0.088384  0.209730 -0.098812  0.610211  0.260313  0.000000 -0.036405
BB      0.517539  0.018272  0.119746 -0.165095  0.252453 -0.036405  0.000000
```

Let us set the parameter $\theta = 0.20$. Then as described above, we create an edge between all pairs of stocks whose correlation coefficient has an absolute value greater than or equal to $\theta$. The resulting adjacency matrix of the graph and graph density are given below:

```
~~~~~~~~~~~~~ Adjacency Matrix ~~~~~~~~~~~~~
[[0. 1. 1. 0. 0. 0. 1.]
 [1. 0. 1. 0. 0. 1. 0.]
 [1. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1. 1.]
 [0. 1. 0. 1. 1. 0. 0.]
 [1. 0. 0. 0. 1. 0. 0.]]
Graph density: 38.10%
```

Finally, we set each nodes' weight to be the stock's annualized return over the same period, rounded to the nearest integer:
```
~~~~~~~~~~~~~ Annualized Returns ~~~~~~~~~~~~~
[ 25.7769583  5.24929993  46.24391841  8.2651049  14.05131557  7.87357035  -19.36790991]

~~~~~~~~~~~~~ Weights ~~~~~~~~~~~~~
[ 26.   5.  46.   8.  14.   8. -19.]
```

This is the resulting weighted market correlation graph with node weights displayed in the parentheses:

<img src="images/small graph.png" width="400"/>

Recall our question from earlier: *Given a set of assets in which we could potentially invest in, how do we identify a subset of uncorrelated assets?* Using the graph above, this translates to finding a set of nodes such that no two nodes are connected. It would also be nice to select nodes such that the sum of their weights is maximized since this would be the "most preferrable" to the investor. The problem of finding such a set of nodes is exactly the *Maximum Weighted Independent Set* problem!

## Maximum Weighted Independent Set

A set of nodes in a (undirected) graph is said to be *independent* if no two nodes in the set are adjacent to each other. For example, consider the sets below on the same graph:

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![Example of independent sets](<images/independent_graph_example.svg>) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![Example of dependent sets](<images/dependent_graph_example.svg>) |
| :-: | :-: |
| Independent Sets | Non-Independent (Dependent) Sets |

The six graphs on the left are independent sets, where the nodes selected in each set are colored in green. Note that an empty set (top left) is independent since the condition for independence is vacuously true. On the other hand, the six graphs on the right are non-independent (dependent) sets, where the nodes selected in each set are colored in red. Any edges within each set are also highlighted in red since they violate the condition for independence.

Given a graph where each node is *weighted* define the *weight* of a set of nodes to be the sum of their individual weights. The Maximum Weighted Independent Set (MWIS) problem aims to find the independent set with the greatest weight.


## Problem Formulation Using QUBO

We can solve the MWIS Problem in TitanQ by formulating it as a Quadratic Unconstrained Binary Optimization (QUBO) problem. Since the variable in question is deciding which nodes to select in the independent set, we can naturally model this as a vector of binary variables. We define
```math
X = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} ~ \text{ where each } x_i \in \{0, 1\}
```
where $x_i = 0$ indicates that node $i$ *is not* selected for set, and $x_i = 1$ indicates that node $i$ *is* selected for the set. $n \in \mathbb{Z}^+$ is the number of nodes in the graph. 

Next, we define the *adjacency matrix* or the $J$-matrix of the graph to be an $n \times n$ matrix whose entries are given by
```math
J_{ij} = 
\begin{cases}
    0 \text{ if node $i$ and node $j$ are not connected} \\
    1 \text{ if node $i$ and node $j$ are connected}
\end{cases}
```
for all $1 \leq i, j \leq n$. Note that since the graph is undirected, then $J$ is symmetric. Moreover, its diagonal entries are 0.

Finally, we define the *bias vector* or the $h$-vector to simply be a vector of length $n$ containing the weights of the nodes:
```math
h = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_n \end{bmatrix}
```
where each $w_i$ is the weight of node $i$. 

Then the objective function to minimize is
```math
\mathcal{H}=
A * \left(\sum_{i=1}^n\sum_{j=1}^n x_ix_j\right)
- B * \left(\sum_{i = 1}^n w_ix_i\right)
= A \left(x^TJx\right) - B \left(h^Tx\right)
```
where $A \geq 0$ and $B \geq 0$ are both hyperparameters that must be tuned. The $A$ term punishes the selection of nodes that are connected by an edge, and the $B$ term maximizes the total weight among selected nodes. In general, the hyperparameters should be set with $A > B$ to increase the likelihood of generating a set that is independent.

If we wish to find the maximum independent set for an unweighted graph (MIS Problem), then we can set $h = [1, 1, \ldots, 1]^T$ (all node weights equal to 1) and so the $B$ term will simply be the number of nodes selected, multiplied by the hyperparameter $B$.

## Hyperparameter Tuning

The examples make use of different hyperparameters, some of which are used in the *QUBO* formulation and others used to tune the TitanQ solver.

The hyperparameters used in the *QUBO* are the following:

- $A$ = Coefficient to penalize the selection of nodes that are connected.

- $B$ = Coefficient to incentivize the selection of nodes with greater weights.

The hyperparameters used to tune the TitanQ solver are the following:

- *beta* = Scales the problem by this factor (inverse of temperature). A lower *beta* allows for easier escape from local minima, while a higher *beta* is more likely to respect penalties and constraints.

- *coupling_mult* = The strength of the minor embedding for the TitanQ specific hardware.

- *timeout_in_secs* = Maximum runtime of the solver in seconds.

- *num_chains* = Number of parallel chains running computations within each engine.

- *num_engines* = Number of independent parallel problems to run. More engines increases the probability of finding an optimal solution.

> **Small changes in these hyperparameters can have a considerable impact on the quality of the solution.**

## TitanQ Example 

Recall the market correlation graph we constructed earlier for the set of seven stocks.

<img src="images/small graph.png" width="400"/>

We can formulate this as an MWIS problem by setting the adjacency matrix as the $J$-matrix, and the node weights as the bias vector $h$:
```math
J =
\begin{bmatrix}
    0 & 1 & 1 & 0 & 0 & 0 & 1 \\
    1 & 0 & 1 & 0 & 0 & 1 & 0 \\
    1 & 1 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 & 1 \\
    0 & 1 & 0 & 1 & 1 & 0 & 0 \\
    1 & 0 & 0 & 0 & 1 & 0 & 0
\end{bmatrix}
\qquad\qquad
h =
\begin{bmatrix} 26 \\ 5 \\ 46 \\ 8 \\ 14 \\ 8 \\ -19 \end{bmatrix}
```

Looking at the graph, we can see by inspection that the maximum weighted independent set is [TSLA, GIS, MCD]. We can also obtain this solution by using the TitanQ SDK:
```
--- RESULTS ---
     x: [0. 0. 1. 1. 1. 0. 0.]
stocks: ['TSLA', 'GIS', 'MCD']
energy: [-0.6799999475479126]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The solution is valid! The corresponding set is independent.
```


A full step-by-step use of the TitanQ SDK to solve this MWIS problem can be found in the Jupyter notebook *example.ipynb*. The notebook uses functions from *stock_utils.py* to construct the correlation and adjacency matrix, and inputs all parameters into TitanQ.

Some larger examples can be found in the *instances* folder, which contains market indices such as Dow Jones, Nasdaq-100 and S&P 500. Note that some of these indices have some constituents removed since they are not tracked by the *yfinance* library.

To run the notebook the following package is required:

- TitanQ SDK

The rest of the required packages are listed in *requirements.txt* and can be installed using pip:

```bash
pip install -r requirements.txt
```

## License

Released under the Apache License 2.0. See [LICENSE](../LICENSE) file.