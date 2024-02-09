# README

#### This project is an example of solving a JSSP using the TitanQ SDK.
--------------------------------------------------------------------------------


This example contains the following sections: 

- Problem Definition

- Problem Instance Format

- Problem Formulation Using QUBO 

- Hyperparameter Tuning

- Solution Visualization

- How to Run

- License

## Problem Definition

[JSSP](https://en.wikipedia.org/wiki/Job-shop_scheduling) (Job Shop Scheduling Problem) is an optimization problem where the goal is to execute a given number of jobs across a given number of machines within the least possible time. A job consists of a series of operations, each with a specific processing time.

The solution must also fulfill the following constraints:

- Operations must be executed in sequential order

- Each machine can execute only one operation at a time

## Problem Instance Format

The input provides information regarding the number of jobs, the number of machines, and the processing time for each operation. The input file is formatted as follows:

````
n   m
job_1_machine_1 job_1_machine_1_p_1 ... job_1_machine_m job_1_machine_m_p_m
job_2_machine_1 job_2_machine_1_p_1 ... job_2_machine_m job_2_machine_m_p_m
        .               .                      .                .          
        .               .                      .                .          
job_n_machine_1 job_n_machine_1_p_1 ... job_n_machine_m job_n_machine_m_p_m
````

where:

- *n* represents the number of jobs

- *m* represents the number of machines

- *job_n_machine_m* represents the machine id in which the *m*th operation for the *n*th job is executed

- *job_n_machine_m_p_m* represents the processing time of the *m*th operation for the *n*th job


The following is an example of an input file for a JSSP with 3 jobs and 3 machines:

```
3   3
1   1   2   2   0   3
2   2   0   1   1   2
0   3   1   3   2   1
```
The first row :

```
3   3
```
depicts the number of jobs (3), and number of available machines (3) respectively.

After the first row, each subsequent row represents a job. For this particular example, each job contains 3 operations running on the 3 different machines. The row corresponding to the first job is highlighted below:

```
1   1   2   2   0   3
```

The first job contains 3 operations:
- The first operation runs on machine id 1 and requires 1 unit of time.
- The second operation runs on machine id 2 and requires 2 units of time.
- The third operation runs on machine id 0 and requires 3 units of time.


## Problem Formulation Using QUBO

In this example, we map the JSSP to a QUBO (*Quadratic Unconstrained Binary Optimization*) problem.

The following expression is the hyperparameterized *QUBO* formulation:

```math
\begin{align}
\mathcal{H}=
\quad&A\sum_i\left(\sum_t x_{i,t}-1\right)^2\\
\quad&+B\sum_m\left(\sum_{(i,t,k,t')\in R_m}x_{i,t}x_{k,t'}\right)\\
\quad&+C\sum_n\left(\sum_{\substack{k_{n-1} < i < k_n\\ t + p_i > t'}} x_{i,t}x_{i+1,t'}\right) \\
\quad&+D\sum_{i \in F} \sum_t tx_{i,t} \\
\quad&+E\sum_{i \in L} \sum_t tx_{i,t}
\end{align}
```

where $x_{i,t}$ is a binary variable such that:

```math
\begin{align}
x_{i,t}=
\begin{cases}
1,\quad\text{operation }i\text{ starts at time }t\\
0,\quad\text{otherwise}
\end{cases}
\end{align}
```

The terms preceded by the hyperparameters *A*, *B*, and *C* affect the constraints embedded into the objective function of the problem.

The terms preceded by the hyperparameters *D* and *E* represent the terms within the objective function to minimize.


$R_m$ represents the set of operations executed by the machine $m$. That is $R_m = A_m \cup B_m$ where:

```math
\begin{align}
&A_m=\{(i,t,k,t'):(i,k)\in\mathcal{I}_m\times\mathcal{I}_m, \quad i \neq k, t \ge 0,T \ge t',0< (t'-t) < p_i\}\\
&B_m=\{(i,t,k,t'):(i,k)\in\mathcal{I}_m\times\mathcal{I}_m,\quad i < k , t = t',p_i > 0,p_j > 0\}\\
&\mathcal{I}_m =\{i: \textit{operation \textit{i} being executed on the machine m}\}
\end{align}
```

$F$ represents the set of all first operations within each job.

$L$ represents the set of all the last operations within each job.

## Hyperparameter Tuning

Various hyperparameters can be tuned throughout the *QUBO* formulation and the TitanQ solver.

The hyperparameters throughout the *QUBO* formulation are explained below:

- The hyperparameters *A*, *B*, and *C* are used to tune the effect of the constraints embedded into the objective function of the problem:

    - A is used for incentivizing each operation to be executed exactly once throughout the entire process.

    - B is used for incentivizing each machine to only execute up to one operation at any given time.

    - C is used for incentivizing the schedule to follow the order of execution for each operation within a particular job specified from the input file. 

- The hyperparameters *D* and *E* are used to tune the effect of terms within the objective function to minimize:

    - D is used for incentivizing machines to perform an operation at the very beginning of the schedule.

    - E is used for incentivizing the machines to complete all jobs within the shortest possible time.

> [!TIP]
> It is typically preferred to have relatively higher values for the hyperparameters *A*, *B*, and *C*, compared to the hyperparameters *D* and *E* to generate valid solutions consistently.


The hyperparameters used to tune the TitanQ solver are the following:

- *beta* = Scales the problem by this factor (inverse of temperature). A lower *beta* allows for easier escape from local minima, while a higher *beta* is more likely to respect penalties and constraints.

- *coupling_mult* = Strength of the minor embedding for the titanQ specific hardware.

- *timeout_in_secs* = Maximum runtime of the solver in seconds.

- *num_chains* = Number of parallel runs executed by the solver. A larger number of parallel runs generally leads to higher quality solutions.

<!-- > [!WARNING]  
> Small changes on these hyperparameters can have an impact on the quality of the solution. Use wisely.
> Preferably keep these hyperparameters in their default values -->

## Solution Visualization

The solution produced by the TitanQ SDK can be visualized with a GANTT chart where each color represents a specific job and each row represents a specific machine. This can generated with the utility function below:

````python
utils.draw_solution(problem,solution,x_max=max_time)
````

## How to Run

A full example is demonstrated in the jupyter notebook *example.ipynb*.

The following package is required:

- TitanQ SDK

The rest of the requirements are listed under *requirements.txt* and can be installed using pip with the following command:

```bash
pip install -r requirements.txt
```

Setting credentials to access the TitanQ SDK is required to run this example. These credentials can be set by creating a file ```.env``` from the file ```.env.example```:

```
TITANQ_DEV_API_KEY = "Your API key"
AWS_ACCESS_KEY = "Your Access key"
AWS_SECRET_ACCESS_KEY = "Your secret access key"
```

## License

Released under the Apache License 2.0. See [LICENSE](../LICENSE) file.
