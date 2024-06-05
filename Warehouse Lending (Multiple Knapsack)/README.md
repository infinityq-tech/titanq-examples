# Warehouse Lending (Multiple Knapsack)

#### This repository contains an example on how to solve the assignment of loans to warehouses while minimizing cost using the TitanQ SDK.

---------------------------------------------

## Table of Contents
- Problem Definition

- Objective

- Mathematical Formulation

- Mixed-Integer Linear Program

- How to Run

- License

----------------------------------------

### Problem Definition

Warehouse lending is a way for a bank to provide loans without using its own capital. Financial institutions provide warehouses lines of credit to mortgage lenders, and the lenders must repay the financial institution. A bank handles the application and approval of a loan and passes the funds from the warehouse lender to a creditor in the secondary market. The bank receives funds from the creditor to pay back the warehouse lender and profits by earning points and original fees.

### Objective

Given a set of warehouses and loans, we would like to know what is the best strategy to take loans from warehouse lenders while minimizing the cost of getting these loans.

### Mathematical Formulation

We define for this problem a binary variable $x_{ij}$ such that:
- $x_{ij} = 1$ if the loan $i$ is assigned to the warehouse $j$.
- $x_{ij} = 0$ otherwise.

The problem defines a set of constraints that represents a feasible solution for the problem:
- A loan can only be assigned to a warehouse that has the same pools.
    - E.g. new_loans[0]= { "value": 10.0, "pools": ["quebec", "blockchain_startup"]} can only be assigned to a warehouse that has a 'quebec' AND 'blockchain_startup' pools
- A loan can only be assigned to one warehouse at a time.
- The loans should respect the pool limit of the warehouse.

The objective is to minimize the total cost of all loans since taking a loan from a certain warehouse generates a certain cost.

The value of a loan is the numerical amount specified by the 'value' key from the loan dictionary.

### Mixed-Integer Linear Program

In this section, we will formulate the problem definition into a objective function and set of constraints as follows:
- Objective : $\sum_i s_i\sum_j value_i\cdot x_{ij}$
- Constraints:
  - A loan can only be assigned to a warehouse that has the same pools:

    - With $`\text{valid\_loan\_wh}`$ being the valid assignment of loans to warehouses such that the loan should share the same pools with the warehouse.
  - A loan can only be assigned to one warehouse at a time:

    $`\sum_{\forall j \in \text{valid\_loan\_wh[i]} } x_{i,j} = 1 \,\,\, \forall i \in [0 ,\cdots , num\_new\_loans]`$

  - The loans should respect the pool limit of the warehouse:
  
    $`\dfrac{value_{j,k} + \sum_{\forall i \in \text{valid\_wh\_loan[j]}} value_{i} \times x_{i,j} }{total\_loans\_wh_{j} + \sum_{\forall i \in \text{valid\_wh\_loan[j]}} value_{i} \times x_{i,j} } < limit_{j,k} \,\,\, \forall j \in [0 ,\cdots , num\_new\_loans], \,\, k \in \text{pools[j]}`$

    - $`value_{j,k}`$ represents the value of the pool k of the warehouse $j$.
    - $`value_i`$ represents the value of the loan.

### How to Run

A full example is demonstrated on the notebook [example.ipynb](example.ipynb).

The requirements are listed under *requirements.txt* and can be installed using pip with the following command:

```bash
pip install -r requirements.txt
```
### License

Released under the Apache License 2.0. See [LICENSE](../LICENSE) file.