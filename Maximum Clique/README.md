# README

**This project gives examples for solving the maximum clique problem using quadratic constraints and the TitanQ SDK.**

## Maximum Clique Problem
Let $G=(V,E)$ be an unweighted, undirected graph. A **clique** of $G$ is a subset $C\subseteq V$ such that for all $u,v\in C$, we have $uv\in E$. In other words, a clique is a complete subgraph of $G$. A maximum clique is a clique such that no clique with more elements exists. The size of a maximum clique is denoted $\omega(G)$ [1].

Finding $\omega(G)$ for an arbitrary graph $G$ is NP-complete. However, we can use TitanQ to find large cliques in $G$ by framing it as an optimization problem.

## Problem Formulation Using Quadratic Constraints
Suppose $G=(V,E)$, where $V=\{v_1,\dots,v_N\}$. We can identify a subset $C\subseteq V$ with a vector $\mathbf{x}=[x_1 \cdots x_N]^T\in \\{0,1\\}^N$ where $x_i = 1$ whenever $v_i\in C$. The size of $C$ is then

$$|C|=\sum_{i=1}^N x_i=\mathbf{1}^T_{N\times 1}\mathbf{x},$$

where $\mathbf{1}_{N\times 1}$ is a column vector of $N$ ones.

We also want to assert that $C$ is a valid clique. That is, whenever $v_i,v_j\in C$ such that $x_ix_j=1$, we need $v_iv_j\in E$. Let $\mathbf{A}=(a_{ij})$ be the adjacency matrix for $G$. To say $v_iv_j\in E$ is the same as $a_{ij}=1$. Then the clique condition can be formulated as $x_ix_j=a_{ij}x_ix_j$ for $i\neq j$. For this to be true for all $i,j$, $i\neq j$, we must have

$$\sum_{i=1}^N\sum_{j=1}^N (1-a_{ij})x_ix_j-\sum_{i=1}^N x_i^2=0$$

or as a matrix product

$$\mathbf{x}^T(\mathbf{1}_{N\times N}-\mathbf{A}-\mathbf{I})\mathbf{x}=0$$

where $\mathbf{1}_{N\times N}$ is an $N\times N$ matrix of ones. This gives a quadratic constraint that our model must satisfy. Hence the optimization problem to solve is:

$$\max_{\mathbf{x}\in\\{0,1\\}^N} \mathbf{1}\_{N\times 1}^T\mathbf{x}\quad\text{subject to}\quad \mathbf{x}^T(\mathbf{1}_{N\times N}-\mathbf{A}-\mathbf{I})\mathbf{x}=0.$$

### Equivalent Formulation as a Maximum Independent Set
Finding a maximum clique in $G$ is equivalent to finding a maximum independent set in the complement $\overline{G}$. We can also use TitanQ to solve this problem, for example, a QUBO formulation is given in the [portfolio optimization example](../Portfolio%20Optimization).

## Instances
The included test instances were collected from CSPLib [2]. The optimal values for the included instances are provided below:

| Instance | Number of Vertices ($N$) |  $\omega(G)$ |
| :--------| :---: | :----------:|
| C125.9.clq | 125 |  34 |
| brock200_2.b | 200 | 12 |
| DSJC500_5 | 500 | 13 |

## Referneces
[1] Weisstein, E. W. Clique. MathWorld A Wolfram Web Resource. https://mathworld.wolfram.com/Clique.html

[2] McCreesh C. Problem 074: Maximum Clique. CSPLib: A problem library for constraints. http://www.csplib.org/Problems/prob074

## License

Released under the Apache License 2.0. See [LICENSE](../LICENSE) file.
