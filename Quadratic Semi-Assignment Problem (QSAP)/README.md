# README

#### This project is an example of solving the QSAP using the TitanQ SDK.

## Table of Contents:
- Problem Definition
- Input File Format
- Filling the Input Files
    - Flow matrix
    - Production cost matrix
    - Distance matrix
- Example: Adding a new location

## Problem Definition

The [QSAP](https://www.researchgate.net/profile/Leonidas-Pitsoulis/publication/249595695_Quadratic_Semi-assignment_Problem/links/574fdf2308aef199238eff1b/Quadratic-Semi-assignment-Problem.pdf) (Quadratic Semi-Assignment Problem) is a specific type of combinatorial optimization problem that arises in various fields including transportation logistics and electronic PCB design. The objective of the QSAP is to assign each element of a set to a unique element of another set to either minimize the total cost or maximize the total profit, subject to certain constraints. The term "semi-assignment" indicates that not all elements need to be assigned, allowing for flexibility in the solution space. The cost or profit associated with each assignment is typically represented by a quadratic function of the assignments, leading to the "quadratic" portion of the QSAP.

## Input File Format

The example Jupyter Notebook for QSAP uses three CSV files as inputs:

- flow_matrix.csv
- production_cost_matrix.csv
- distance_matrix.csv

The **flow matrix** describes the dependency of materials. It acts as a dependency graph to let the program know that material 1 and 2 are both dependent on each other.
The n x n matrix is symmetric where n is the total number of materials. The elements of the flow matrix can only take the values 1 or 0. A value of 1 indicates that the corresponding materials are dependent on each other. If not, a value of 0.

The **production cost matrix** lists the cost it takes for a specific facility at a specific location to finish a certain material. The matrix has size n x m where n is the number of materials and m is the number of facilities.

The **distance matrix** lists the distance that it takes for a material to be transferred from facility A to facility B. The m x m matrix is symmetric where m is the total number of facilities.

## Example: Adding a new location

Suppose we start working with a new location that can provide one extra facility. We need to add the information about the facility and the location to the production cost matrix and the distance matrix. The flow matrix can stay untouched since the list of tasks remains unchanged.

### Modifying the production cost matrix

We can add the new facility to the production cost matrix in a few simple steps. First, we add a new column to the end of the existing matrix to make room for the new facility. Next, we determine what materials the facility is capable of processing. That information helps us to find the relevant rows in the matrix that we need to edit. Let's assume the facility can handle Material 1 and Material 2. We look at our list of materials and check which positions Material 1 and Material 2 hold within the overall list. Then we use those positions as our indices for the rows of the production cost matrix and fill the last element of those rows with the cost it takes the new facility to finish the materials. Lastly, we fill all remaining elements in the column with a large negative number (e.g, -1000) to avoid invalid assignments of materials to this facility. 

### Modifying the distance matrix

The first step in modifying the distance matrix is to calculate the cost for shipping and customs for transferring materials from the new locations to all the other locations in our system. Then we can use this information to add a new row in our distance matrix for the new machine and input each cost in the corresponding order. Next, we have to add a new column at the end of the matrix to also cover the cost it takes to transport materials from the original locations to the new location. The final matrix element is set to a large factor (in our example 1000) to incentivize spreading materials across multiple facilities.