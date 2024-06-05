# README

#### This project is an example of solving a JSSP using the TitanQ SDK.
--------------------------------------------------------------------------------


This example contains the following sections: 

- Problem Definition

- Problem Instance Format

- Problem Formulation Using MILP

- Hyperparameter Tuning

- Solution Visualization

- How to Run

- License

## Problem Definition

The [JSSP](https://en.wikipedia.org/wiki/Job-shop_scheduling) (Job Shop Scheduling Problem) is an optimization problem where the goal is to execute a given number of jobs across a given number of machines within the least possible time. A job consists of a series of operations, each with a specific processing time.

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


## Problem Formulation Using MILP 

In this example, we map the JSSP to a MILP (*Mixed-Integer Linear Programming*) as the following model:

```math
\begin{array}{lll}
\min & \sum_{i \in L} x_i & \\
\mathrm{s.t.} & 0 \leq x_{i} \leq max_{time}, & \forall i \in I\\
& x_{i} \geq x_{i-1}+p_{i-1}, & \forall i, k_{l-1}+1 \leq i \leq k_l, \forall l, 1 \leq l \leq N \\
& x_{i} \geq x_{j}+p_{j}-V \cdot z_{i j}, & \forall i,j \in I\times I , i < j\\
& x_{j} \geq x_{i}+p_{i}-V \cdot (1-z_{i j}), & \forall i,j \in I\times I , i < j\\
& H \cdot (y_{ml_ji}-1) + x_{i} + p_i\leq l, & \forall m \in M, \forall i \in I_m,  \forall (l,u) \in B_{m}\\
& u \leq x_i + H \cdot y_{ml_ji}, & \forall j \in \{1,\cdots,N_{bm}\}\\
& z_{i j} \in\{0,1\}, & \forall i,j \in I\times I
\end{array}
```

where :

- $x_{i}$ is an integer variable indicating the start time of the task $i$.
- $z_{ij}$ is a binary variable. 
  - $z_{ij} = 1$ indicates that the task $i$ precedes the task $j$.
  - $z_{ij} = 0$ otherwise.
- $y_{ml_{j}i}$ is a binary variable.
  - $y_{ml_{j}i} = 1$ if the task $i$ precedes the $j$ th blocked time slot on the machine $m$.
  - $y_{ml_{j}i} = 0$ otherwise.
- $I$ is the set of tasks.
- $M$ is the set of machines.
- $N$ is the number of jobs.
- $N_T$ is the number of tasks.
- $N_M$ is the number of machines.
- $N_{bm}$ is the number of blocked time slots on the machine $m$.
- $I_m$ is the set of tasks running on machine $m$.
- $p_i$ is the duration of the task $i$.
- $k_{l-1} < i \leq k_l$ indicates the task indices within the job $j$.
- $`B_m = \{(l_1,u_1),\cdots,(l_{N_{bm}},u_{N_{bm}})\}`$ is the set of blocked time slots on the machine $m$.
- $L$ represents the set of all the last operations within each job.

## Hyperparameter Tuning

The hyperparameters used to tune the TitanQ solver are the following:

- *beta* = Scales the problem by this factor (inverse of temperature). A lower *beta* allows for easier escape from local minima, while a higher *beta* is more likely to respect penalties and constraints.

- *timeout_in_secs* = Maximum runtime of the solver in seconds.

- *num_chains* = Number of parallel runs executed by the solver. A larger number of parallel runs generally leads to higher quality solutions.

## Solution Visualization

The solution produced by the TitanQ SDK can be visualized with a GANTT chart where each color represents a specific job and each row represents a specific machine. This can generated with the utility function below:

````python
utils.plot_schedule(
    assignment,
    schedule,
    available_time_slots,
    machine_names,
    unit="days"
)
````

## How to Run

A full example is demonstrated by showcasing three JSSP instances of different sizes:
- Small: 2 jobs and 2 machines (example_small.ipynb).
- Medium: 6 jobs and 6 machines (example_medium.ipynb).
- Large: 10 jobs and 5 machines (example_large.ipynb).

The requirements are listed under *requirements.txt* and can be installed using pip with the following command:

```bash
pip install -r requirements.txt
```
## License

Released under the Apache License 2.0. See [LICENSE](../LICENSE) file.
