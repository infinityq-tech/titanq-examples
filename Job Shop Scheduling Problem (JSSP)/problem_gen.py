# Copyright (c) 2024, InfinityQ Technology Inc.

import numpy as np
from utils import *

def generate_variable_bounds(jobs, handoffs, Nx, Nz, Ny, max_start_time):
    """ Generates the variable bounds for each variable.

    Args:
        Nx (int): Number of variables representing start times.
        Nz (int): Number of variables to handle machine overlapping constraint.
        Ny (int): Number of variables to handle blocked time overlapping constraint.
        max_start_time (int): The latest possible start time for a task.

    Returns:
        Numpy.array[Numpy.float32, Numpy.float32]: List of upper and lower bounds for each variable.
    """

    N = Nx + Nz + Ny
    variable_bounds = np.zeros((N, 2), dtype=np.float32)

    for i in range(Nx):
        variable_bounds[i][1] = max_start_time

    for job_id, tasks in jobs.items():
        num_tasks = len(tasks)

        min_start_time = 0
        for task_id in range(num_tasks):
            global_task_id = get_global_index_of_task(
                jobs, job_id, task_id)
            variable_bounds[0 + global_task_id][0] = min_start_time

            # Don't attempt to grab non-existent handoff for last task in a job
            if task_id <= num_tasks - 2:
                # Next task can't start earlier than the absolute earliest possible start time of the previous task
                min_start_time += jobs[job_id][task_id][1] + \
                    handoffs[jobs[job_id][task_id][0]][jobs[job_id][task_id + 1][0]]

    for j in range(Nx, N):
        variable_bounds[j][0] = 0
        variable_bounds[j][1] = 1

    return variable_bounds

def generate_variable_types(Nx, Nz, Ny):
    """ Generates the list of variable types.

    Args:
        Nx (int): Number of variables representing start times.
        Nz (int): Number of variables to handle machine overlapping constraint.
        Ny (int): Number of variables to handle blocked time overlapping constraint.

    Returns:
        List[str]: The list of variable types for each variable.
    """

    N = Nx + Nz + Ny
    variable_types = []

    for i in range(Nx):
        variable_types.append("i")

    for j in range(Nx, N):
        variable_types.append("b")

    return variable_types


def last_task_for_each_job(jobs: dict):
    """ Generates the list of indices of all last tasks for each job.

    Args:
        jobs (Dict{int:List[(Tuple(int,int))]}): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key.

    Returns:
        List[int]: The list of indices of all last tasks for each job
    """

    l = []
    cumulative = 0
    for _, t in jobs.items():
        l.append(cumulative + len(t) - 1)
        cumulative += len(t)
    return l


def first_task_for_each_job(jobs: dict):
    """ Generates the list of indices of all first tasks for each job.

    Args:
        jobs (Dict{int:List[(Tuple(int,int))]}): Dictionary where the key is the job number
        and the value is a list of execution times of the tasks of the corresponding job key.

    Returns:
        List[int]: The list of indices of all first tasks for each job.
    """

    f = []
    cumul = 0
    for _, t in jobs.items():
        f.append(cumul)
        cumul += len(t)
    return f


def generate_weights_bias(jobs, num_variables):
    """ Generates the weight matrix W and bias vector b of the objective function:
            -1/2 * x^T.W.x - b.x

    Args:
        jobs (Dict{int:List[(Tuple(int,int))]}): Dictionary where the key is the job number
            and the value is a list of execution times of the tasks of the corresponding job key.
        num_variables (int): The number of variables of the problem.
        A (float): The hyperparameter used for incentivizing machines to finish tasks as soon as possible in the schedule.

    Returns:
        Numpy.array[Numpy.float32,Numpy.float32]: The weight matrix of the objective function.
        Numpy.array[Numpy.float32]: The bias vector of the objective function.
    """

    W = np.zeros((num_variables, num_variables), dtype=np.float32)
    b = np.zeros(num_variables, dtype=np.float32)
    L = last_task_for_each_job(jobs)
    for i in L:
        b[i] += 1

    return W, b
