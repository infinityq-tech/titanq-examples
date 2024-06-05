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


def generate_constraint_weights_bounds(
    jobs,
    num_variables,
    Nx,
    Nz,
    V,
    handoffs,
    num_machines,
    available_time_slots,
    H,
    
):
    """ Generates the constraint weights A and constraint bounds (c^l,c^u) of the constraint:
            c^l <= A.x <= c^u 

    Args:
        jobs (Dict{int:List[(Tuple(int,int))]}): Dictionary where the key is the job number
            and the value is a list of execution times of the tasks of the corresponding job key.
        num_variables (int): The number of variables of the problem.
        Nx (int): Number of variables representing start times.
        Nz (int): Number of variables to handle machine overlapping constraint.
        V (int): Hyperparameter used to enforce not having two tasks running on the same machine at a time.
        handoff (Numpy.array[int]): The Supplier/Machine Transfer Time Matrix.
        num_machines (int): Number of available machines.
        available_time_slots (Numpy.array[int,int]): The available time slots of the machines.
        H (int): The value used to enforce not having tasks running on a blocked time slot.
        
    Returns:
        Numpy.array[Numpy.float32,Numpy.float32]: The matrix A with constraint coefficients.
        Numpy.array[Numpy.float32,Numpy.float32]: The vector of lower and upper bounds for each constraint.
    """
    
    # List of tasks
    tasks = [item for sublist in jobs.values() for item in sublist]  
    # Helper array
    z_idx = [(i, j) for i in range(Nx) for j in range(Nx) if tasks[i][0] == tasks[j][0]]

    A = np.empty((0, num_variables), dtype=np.float32)
    c = np.empty((0, 2), dtype=np.float32)

    row = 0

    # Constraint #1: Precedence constraint
    col = 0
    for job, tasks in jobs.items():
        num_tasks = len(tasks)
        for i in range(1, num_tasks):
            c = np.append(c, np.zeros((1, 2), dtype=np.float32), axis=0)
            A = np.append(A, np.zeros(
                (1, num_variables), dtype=np.float32), axis=0)

            A[row][col+i] = 1
            A[row][col+i-1] = -1
            c[row][0] = jobs[job][i-1][1] + \
                handoffs[jobs[job][i-1][0]][jobs[job][i][0]]
            c[row][1] = np.NaN
            row += 1
        col += num_tasks

    # Constraint #2: Machine overlapping
    for job_1, tasks_1 in jobs.items():
        num_tasks_1 = len(tasks_1)

        for job_2, tasks_2 in jobs.items():
            num_tasks_2 = len(tasks_2)

            for i in range(num_tasks_1):
                for j in range(num_tasks_2):
                    _i = get_global_index_of_task(jobs, job_1, i)
                    _j = get_global_index_of_task(jobs, job_2, j)

                    machine_i = jobs[job_1][i][0]
                    machine_j = jobs[job_2][j][0]

                    if (_i < _j) and (machine_i == machine_j):
                        c = np.append(c, np.zeros(
                            (1, 2), dtype=np.float32), axis=0)
                        A = np.append(A, np.zeros(
                            (1, num_variables), dtype=np.float32), axis=0)

                        A[row][_i] = 1
                        A[row][_j] = -1

                        A[row][Nx + z_idx.index((_i, _j))] = V

                        c[row][0] = jobs[job_2][j][1]
                        c[row][1] = np.NaN

                        row += 1

    # Constraint #3: Machine overlapping
    for job_1, tasks_1 in jobs.items():
        num_tasks_1 = len(tasks_1)

        for job_2, tasks_2 in jobs.items():
            num_tasks_2 = len(tasks_2)

            for i in range(num_tasks_1):
                for j in range(num_tasks_2):
                    _i = get_global_index_of_task(jobs, job_1, i)
                    _j = get_global_index_of_task(jobs, job_2, j)

                    machine_i = jobs[job_1][i][0]
                    machine_j = jobs[job_2][j][0]

                    if (_i < _j) and (machine_i == machine_j):
                        c = np.append(c, np.zeros(
                            (1, 2), dtype=np.float32), axis=0)
                        A = np.append(A, np.zeros(
                            (1, num_variables), dtype=np.float32), axis=0)

                        A[row][_i] = -1
                        A[row][_j] = 1

                        A[row][Nx + z_idx.index((_i, _j))] = -V

                        c[row][0] = jobs[job_1][i][1] - V
                        c[row][1] = np.NaN

                        row += 1
                            
    # Constraint #4: Blocked time slots
    flatten_tasks = [item for row in jobs.values() for item in row]
    y_counter = 0
    for m_idx in range(num_machines):
        # Find the indices where -1 occurs
        indices = np.where(available_time_slots[m_idx] == -1)[0]

        # Find the groups of -1 indices
        groups = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)

        # Filter out groups with only one element
        time_slots = [(group.tolist()[0],group.tolist()[-1]) 
                      for group in groups if len(group) >= 1]
        I_m = [idx for idx,(machine,_) in enumerate(
            flatten_tasks) if machine==m_idx]
        for i in I_m:
            for j,(l,u) in enumerate(time_slots):
                
                c = np.append(c, np.zeros((1, 2), dtype=np.float32), axis=0)
                A = np.append(A, np.zeros(
                    (1, num_variables), dtype=np.float32), axis=0)

                A[row][i] = 1
                A[row][Nx+Nz+y_counter] = H

                c[row][1] = (l-1 if l>0 else l) - flatten_tasks[i][1] + H
                c[row][0] = np.NaN

                row += 1

                c = np.append(c, np.zeros((1, 2), dtype=np.float32), axis=0)
                A = np.append(A, np.zeros(
                    (1, num_variables), dtype=np.float32), axis=0)

                A[row][i] = 1
                A[row][Nx+Nz+y_counter] = H

                c[row][0] = u + 1
                c[row][1] = np.NaN

                row += 1
                y_counter += 1

    return A, c