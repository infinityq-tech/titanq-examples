# README

#### This project is an example of solving the **Protein Folding Problem** using the TitanQ SDK.

## Table of Contents

- Requirements

- Introduction

- Problem Definition

- Model Generation

- Conversion From a Higher-Order Objective to QUBO

- Solving on TitanQ

- Interpretation of the Solution in Graphical Representation

- References

## Requirements

Python 3.10 is required to run the example. Note that issues may occur if a later version is utilized. The rest of the requirements are listed under `requirements.txt`.

**Note**: Make sure to clone the `quantum-protein-folding` submodule to run the example.
```python
git clone --recurse-submodules
```

## Introduction

The Protein Folding Problem refers to the prediction of the three-dimensional structure of a protein based on its fundamental sequence of amino acids[^2][^3][^4]. There are 20 different amino acids[^5] and proteins consist of one or more chains of amino acids, referred to as polypeptides.

Protein structure has been extensively researched for over a half-century due to its importance in chemistry, biology, and medicine. Many classical algorithms have been developed to tackle this problem. However, they suffer from limited performance and accuracy when confronted with larger problems due to the intrinsic NP-hardness of the problem.

## Problem Definition

The goal of this work is to determine the minimum energy conformation of a protein. Starting with an initial configuration, the protein's structure is optimized to lower the energy. This can be achieved by encoding the protein folding problem into a qubit operator and ensuring that all physical constraints are satisfied as follows:

- penalty_chiral: A penalty parameter used to impose the right chirality[^11].

- penalty_back: A penalty parameter used to penalize turns along the same axis. This term is used to eliminate sequences where the same axis is chosen twice in a row to disallow a chain from folding back into itself.

- penalty_1: A penalty parameter used to penalize local overlap between beads within a nearest neighbor contact.

The problem is modeled on a Tetrahedral lattice[^1][^6][^7] and is formulated as follows:

```math
\begin{align}
H(\mathbf{q}) = H_{gc}(\mathbf{q}_{c}) + H_{ch}(\mathbf{q}_{c}) + H_{in}(\mathbf{q}_{in})
\end{align}
```

where

- $`\mathbf{q}=\{\mathbf{q}_{c}, \mathbf{q}_{in}\}`$

    - $`\mathbf{q}_{c}`$ is a qubit used to describe the conformation.

    - $`\mathbf{q}_{in}`$ is a qubit used to describe the interactions.

- $`H_{gc}`$ is the geometrical constraint term which governs the growth of the primary sequence of amino acids without bifurcations.

- $`H_{ch}`$ is the chirality constraint which enforces the right stereochemistry[^8] for the system.

- $`H_{in}`$ is the interaction energy term of the system. In our case we only consider the nearest neighbor interactions.

The conversion of $`H(\mathbf{q})`$ into an Ising Hamiltonian $`H`$ results in:

```math
\begin{align}
H = \sum_i h_i \sigma_z^i + \sum_{ij} J_{ij} \sigma_z^i \sigma_z^j + \sum_{ijk} K_{ijk} \sigma_z^i \sigma_z^j \sigma_z^k + \sum_{ijkl} L_{ijkl} \sigma_z^i \sigma_z^j \sigma_z^k \sigma_z^l + \sum_{ijklm} M_{ijklm} \sigma_z^i \sigma_z^j \sigma_z^k \sigma_z^l \sigma_z^m
\end{align}
```

where each $`\sigma_z^i`$ represents the Pauli matrix corresponding to the Pauli gate $`Z`$ acting on the qubit $`i`$ and rotating the qubit around the z-axis of the Bloch sphere[^9] by $`\pi`$ radians.

The Hamiltonian $`H`$ can be expressed using the decision variables $`z_i \in \{-1,+1\}`$ in a straightforward way from substituting each $`\sigma_z^i`$ to $`z_i`$[^10] resulting in the following format: 

```math
\begin{align}
H = \sum_i h_i z_i + \sum_{ij} J_{ij} z_i z_j + \sum_{ijk} K_{ijk} z_i z_j z_k + \sum_{ijkl} L_{ijkl} z_i z_j z_k z_l + \sum_{ijklm} M_{ijklm} z_i z_j z_k z_l z_m
\end{align}
```

### Model Generation

The [input data](inputs/mj_matrix.txt) used to generate the model describes the interaction between the 20 different amino acids using the Mayazawa-Jernigan interaction[^12].


### Conversion From a Higher-Order Objective to QUBO
The 5th order Hamiltonian $`H`$  for this problem is first reduced to 2nd order to be solved on TitanQ. One approach that can be used to accomplish this transformation is the decomposition method.

The decomposition to a QUBO model consists of substituting a product of two decision variables with a new variable and adding penalty terms, thus leading to a reduced model. This method uses the following equivalences that must always be held:

```math
\begin{align}
xy = z & & if  & & xy -2xz + 3z = 0\\
xy \neq z & & if &  & xy -2xz + 3z > 0\\
\end{align}
```
The objective is to have the penalty term equal to $`0`$ in order to make the substitution valid.

`ReduceMin()`is an iterative algorithm that reduces the order of a model by one at each iteration until reaching a quadratic model[^13]. The implementation of this algorithm can be found within [utils.py](utils.py).

> [!NOTE]  
> The Hamiltonian $`H`$ should be converted from a Bipolar format to a Binary format first.
> See `transform_polynomial()` in [utils.py](utils.py).

### Solving on TitanQ

The Hamiltonian $`H`$ formulated as a QUBO model can be inserted into TitanQ using the following code snippet:

```python
model = Model(api_key=TITANQ_DEV_API_KEY)

# Construct the problem
model.add_variable_vector('x', len(weights), Vtype.BINARY)
model.set_objective_matrices(weights, bias, Target.MINIMIZE)

# Set hyperparameters and call the solver
num_chains = 2
num_engines = 256
T_min = 1e3
T_max = 1e16
beta = (1.0/np.linspace(T_min, T_max, num_chains)).tolist()

timeout_in_seconds = 5
coupling_mult = 0.5

response = model.optimize(
    beta=beta,
    timeout_in_secs=timeout_in_seconds,
    num_engines=num_engines,
    num_chains=num_chains,
    coupling_mult=coupling_mult
)
```

Note that the `weights` and `bias` are constructed beforehand from the QUBO model previously discussed.

### Interpretation of the Solution in Graphical Representation

The solution returned by TitanQ is a vector with a length equivalent to the number of variables available after performing the decomposition to a QUBO model. The final solution to the Protein Folding Problem is then retrieved from the first $n$ elements of the solution vector, where $n$ is equal to the number of variables of the initial problem.

The graphical visualization generated at the end of the `example.ipynb` notebook is a 3D plot where all the amino acids are represented with the corresponding bonds in the folded conformation.

## References

[^1]: Chandarana, P., Hegade, N. N., Montalban, I., Solano, E., & Chen, X. (2023). Digitized counterdiabatic quantum algorithm for protein folding. Physical Review Applied, 20(1). https://doi.org/10.1103/physrevapplied.20.014024

[^2]: Amino Acids and Proteins. In: Janson LW, Tischler ME. eds. The Big Picture: Medical Biochemistry. McGraw Hill; 2018. The Big Picture: Medical Biochemistry. McGraw Hill; 2018. Accessed 12/22/2021.

[^3]: Medline Plus. Amino acids (https://medlineplus.gov/ency/article/002222.htm). Accessed 12/22/2021.

[^4]: National Human Genome Research Institute. Amino Acids (https://www.genome.gov/genetics-glossary/Amino-Acids). Accessed 12/22/2021.

[^5]: The Twenty Amino Acids: https://www.cryst.bbk.ac.uk/education/AminoAcid/the_twenty.html

[^6]: Tetrahedral molecular geometry: https://en.wikipedia.org/wiki/Tetrahedral_molecular_geometry

[^7]: Crystal structure: https://en.wikipedia.org/wiki/Crystal_structure

[^8]: Stereochemistry: https://en.wikipedia.org/wiki/Stereochemistry

[^9]: Bloch sphere https://en.wikipedia.org/wiki/Bloch_sphere

[^10]: Wang, Z., Hadfield, S., Jiang, Z., & Rieffel, E. G. (2018). Quantum approximate optimization algorithm for MaxCut: A fermionic view. Physical Review. A/Physical Review, A, 97(2). https://doi.org/10.1103/physreva.97.022304

[^11]: Chirality and Stereoisomers: https://chem.libretexts.org/Bookshelves/Organic_Chemistry/Supplemental_Modules_(Organic_Chemistry)/Chirality/Chirality_and_Stereoisomers

[^12]: Miyazawa, S., & Jernigan, R. L. (1996). Residue – Residue Potentials with a Favorable Contact Pair Term and an Unfavorable High Packing Density Term, for Simulation and Threading. Journal of Molecular Biology/Journal of Molecular Biology, 256(3), 623–644. https://doi.org/10.1006/jmbi.1996.0114

[^13]: Boros, E., & Hammer, P. L. (2002). Pseudo-Boolean optimization. Discrete Applied Mathematics, 123(1–3), 155–225. https://doi.org/10.1016/s0166-218x(01)00341-9
