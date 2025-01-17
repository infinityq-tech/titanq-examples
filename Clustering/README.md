# README

#### This project presents examples of using the TitanQ SDK for clustering. It also includes a comparison between Gurobi and K-Means.
--------------------------------------------------------------------------------


Here is an overview:

- Introduction

- Mathematical Formulation

- TitanQ

- License

## Introduction

Clustering is the general process of generating structure within datasets by grouping associated data points with strong similarity. There are many different applications for clustering algorithms including: 

- Image Segmentation 

- Anomaly Detection 

- Medical Imaging

- Social Network Analysis 

- Data Compression 

Clustering can also be applied as an initial step towards solving Vehicle Routing Problems (VRPs). Many established solutions involve clustering a large set of delivery addresses into distinct non-overlapping groups as the first step. Afterwards, an individual Traveling Salesman Problem (TSP) is solved within each previously identified cluster. Clearly, the quality of the initial clusters generated will greatly affect the quality of the overall VRP solution.  

Many different clustering algorithms exist of varying complexity such as K-Means, K-Medoids, and DBSCAN. However, most of these generic algorithms do not allow for the incorporation of additional problem-specific constraints. One example, under the context of the VRP, is to make the number of delivery addresses per vehicle roughly equal such that driver shifts are uniform across the fleet. Basic clustering algorithms, such as the ones mentioned previously, do not provide simple methodologies to incorporate this desired property among generated clusters. Hence, this motivated InfinityQ to develop a novel optimization-based clustering approach.  


## Mathematical Formulation

Clustering is posed as an optimization problem with binary variables in the following mathematical formulation: 
```math
E(x) =  \sum_i \sum_{j>i} \sum_k d_{ij} x_{ik} x_{jk}
```
```math
s.t.   \sum_k x_{ik} = 1 \forall i
```
where:
```math
x_{ik} =
 \begin{cases} 
      1  \text{ if data point $i$ belongs to cluster $k$}\\
      0  \text{ otherwise}
   \end{cases}
```

The objective function is defined to minimize the distance between points within a particular cluster. Note that $d_{ij}$ represents the distance between data point $i$ and data point $j$.  

A set partitioning constraint is also included to ensure that each data point belongs to exactly one cluster. Under the context of the VRP, this refers to each address being assigned to exactly one delivery vehicle.  

Note that the total number of variables in this formulation is $n$ x $k$, where $n$ is the number of data points to cluster and $k$ is the number of desired clusters. Under the context of the VRP, $n$ represents the number of delivery addresses and $k$ represents the total number of delivery vehicles utilized.  

The constraint of having each cluster contain approximately the same number of points is integrated by adding a Lagrangian term to the objective function as shown below:
```math
E_{rebal}(x) =  E(x) + \lambda \sum_k \bigg(\sum_i x_{ik}-k_{avg} \bigg)^2
```
where:
```math
k_{avg} = \frac{\text{Number of Coordinates}}{\text{Number of Clusters}}
```

## TitanQ

A full step-by-step use of the TitanQ SDK for solving the clustering problem can be found in the Jupyter notebook *example.ipynb*. The notebook uses functions from *clustering_utils.py* to calculate performance metrics and to load TitanQ hyperparameters on specific instances.

The hyperparameters used to tune the TitanQ solver are the following:

- *beta* = Scales the problem by this factor (inverse of temperature). A lower *beta* allows for easier escape from local minima, while a higher *beta* is more likely to respect penalties and constraints.

- *coupling_mult* = The strength of the minor embedding for the TitanQ specific hardware.

- *timeout_in_secs* = Maximum runtime of the solver in seconds.

- *num_chains* = Number of parallel chains running computations within each engine.

- *num_engines* = Number of independent parallel problems to run. More engines increases the probability of finding an optimal solution.

> **Small changes in these hyperparameters can have a considerable impact on the quality of the solution.**

To run the notebook the following package is required:

- TitanQ SDK

The rest of the required packages are listed in *requirements.txt* and can be installed using pip:

```bash
pip install -r requirements.txt
```

## License

Released under the Apache License 2.0. See [LICENSE](../LICENSE) file.
