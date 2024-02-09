import math
from copy import deepcopy
from datetime import datetime
from collections import defaultdict
import numpy as np
import plotly.express as px

def read_instance(path: str) -> dict:
    """ Read the JSSP instance from the file path

    Args:
        path (str): The file path of the JSSP

    Returns:
        dict: The jobs dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key
    """

    job_dict = defaultdict(list)
    with open(path) as f:
        f.readline()
        for i, line in enumerate(f):
            lint = list(map(int, line.split()))
            job_dict[i + 1] = [x for x in
                               zip(lint[::2],  # machines
                                   lint[1::2]  # operation lengths
                                   )]
        return dict(job_dict)

def from_sigma_to_solution(sigma, jobs: dict):
    """ Transform any solution for JSSP in the Ising model to a readable solution

    Args:
        sigma (List[int[-1,+1]]): The solution represented in the Ising model
        jobs (dict): jobs (dict): The jobs dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key

    Returns:
        Dict[int:List[int]]: The solution corresponding to the sigma
    """
    solution = {}
    for i, row in enumerate(sigma):
        if 1 in row:
            time = np.where(row == 1)
            time = time[0][0]
            job = get_job_index(jobs, i)
            if job in solution.keys():
                solution[job].append(time)
            else:
                solution[job] = [time]
        else:
            return 0
    return solution

def from_solution_to_sigma(solution:dict,num_nodes,num_rows,num_cols):
    """ Transform any solution to a solution in the Ising model

    Args:
        solution (Dict[int:List[int]]): Dictionary corresponding to the solution of the JSSP where the key is the machine id, 
        and the value is a list of starting times corresponding to each task within a particular job
        num_nodes (int): The number of the nodes of the Ising model
        num_rows (int): The number of rows in the Ising model corresponding to the number of tasks
        num_cols (int): The number of columns in the Ising model corresponding to the max_time 
        for finishing all the tasks

    Returns:
        sigma (List[int[-1,+1]]): The solution represented in the Ising model
    """

    sigma = -np.ones(num_nodes)
    sigma_matrix = np.reshape(sigma, (num_rows, num_cols))
    cumulative = 0
    for _,value in solution.items():
        for i,t in enumerate(value):
            sigma_matrix[cumulative+i][t] = 1
        cumulative+= len(value)
    sigma = np.insert(sigma,0,-1)
    return sigma

def get_max_time(jobs):
    """ Compute the maximum time in order to execute all the tasks of the JSSP

    Args:
        jobs (dict): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key

    Returns:
        int: The maximum time to run all the tasks
    """

    max_time = 0
    for job in jobs.values():
        max_time += sum(a[1] for a in job)
    return max_time


def get_num_machines(jobs: dict):
    """ Compute the number of machines in the JSSP instance

    Args:
        jobs (dict): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key

    Returns:
        int: The number of machines in the JSSP instance
    """

    return max(max(machine for machine, p in tasks) for _, tasks in jobs.items()) + 1


def get_num_tasks(jobs):
    """ Compute the number of tasks in the JSSP instance

    Args:
        jobs (dict): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key

    Returns:
        int: The number of tasks in the JSSP instance
    """

    num_tasks = 0

    for _, value in jobs.items():
        num_tasks += len(value)

    return num_tasks

def get_num_nodes(jobs, max_time):
    """ Compute the number of nodes (spins) in the Ising model of the JSSP instance

    Args:
        jobs (dict): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key
        max_time (int): The maximum time to run all tasks

    Returns:
        int: The number of nodes (spins) in the Ising model
    """

    size = 0
    for _, tasks in jobs.items():
        size += len(tasks)
    return max_time * size

def spin_to_bit(spin):
    """ Transform the spin(-1,+1) to a binary(0,1)

    Args:
        spin (int[-1,+1]): The value of the spin

    Returns:
        int: The value of the spin in the binary format
    """

    return (spin+1)//2

def last_task_for_each_job(jobs:dict):
    """ Generates the list of indices of all last tasks for each job

    Args:
        jobs (dict): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key

    Returns:
        List[int]: The list of indices of all last tasks for each job
    """

    l = []
    cumulative = 0
    for _, t in jobs.items():
        l.append(cumulative + len(t) - 1)
        cumulative += len(t)
    return l

def first_task_for_each_job(jobs:dict):
    """ Generates the list of indices of all first tasks for each job

    Args:
        jobs (dict): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key

    Returns:
        List[int]: The list of indices of all first tasks for each job
    """

    f = []
    cumul = 0
    for _, t in jobs.items():
        f.append(cumul)
        cumul += len(t)
    return f

def compute_total_time(sigma, jobs):
    """ Compute the total time spent given a solution in the Ising model

    Args:
        sigma (List[int[-1,+1]]): The solution represented in the Ising model
        jobs (dict): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key

    Returns:
        int: The total time for the given solution to execute all the tasks
    """

    s = sigma[1:]
    num_rows = get_num_tasks(jobs)
    num_cols = get_max_time(jobs)
    s = np.reshape(s, (num_rows, num_cols))
    total_time = math.inf
    tasks = create_tasks_from_jobs(jobs)
    L = last_task_for_each_job(jobs)
    start = 0
    list_total_time = []
    for j in L:
        col_one = np.where(s[j]==1)[0][0]
        total_time = col_one + tasks[j][1]
        list_total_time.append(total_time)
    for j in range(num_cols):
        index_ones_in_col = [i for i in range(num_rows) if s[i][j] == 1]
        if index_ones_in_col:
            start = j
            break
    return max(list_total_time)-start

def squash_lengths(instance, steps=[4, 7]):
    """ Returns an instance with the same operations, but with
    squashed lengths to [1,2,3,..., len(steps)+1]

    Args:
        instance (dict): instance to be squashed
        steps (list, optional): lengths at which operations
        are qualified as a longer length. Defaults to [4, 7].

    Returns:
        dict: The instance with the same operations but with squashed lengths
    """

    steps.sort()
    steps.append(float('inf'))

    result = deepcopy(instance)

    for operations in result.values():
        for j, operation in enumerate(operations):
            for i, step in enumerate(steps, start=1):
                if operation[1] < step:
                    operations[j] = (operation[0], i)
                    break
    return result


def legal_starting_time(jobs:dict):
    """ Generates a dictionary that indicates at which time each task can start

    Args:
        jobs (dict): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key

    Returns:
        dict: The dictionary where the key is the id of the task 
            and the value is the time from which this task can start
    """

    d = {j : {} for j in jobs.keys()}

    for job,l in d.items():
        s = 0
        for i in range(len(jobs[job])):
            l[i] = s
            s += jobs[job][i][1]
    count = 0
    r = {}
    for job,tasks in d.items():
        for _, time in tasks.items():
            r[count] = time
            count+=1
    return r


def convert_to_datetime(x):
    """ Format the time in a readable format for plotly.express

    Args:
        x (int): The discrete time

    Returns:
        datetime.str: The formatted time
    """

    return datetime.fromtimestamp(31536000 + x * 24 * 3600).strftime("%Y-%m-%d")

def get_result(jobs, solution):
    """ Compute the time spent for the solution

    Args:
        jobs (dict): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key
        solution (dict): Dictionary corresponding to the solution of the JSSP where the key is the machine id, 
        and the value is a list of starting times corresponding to each task within a particular job

    Returns:
        int: The total time spent to finish all tasks
    """
    max_time = 0
    for job, operations in jobs.items():
        max_time = max(max_time, solution[job][-1] + int(operations[-1][1]))
    return max_time

def draw_solution(jobs: dict, solution: dict, x_max=None):
    """Draw the solution using a GANTT chart

    Args:
        jobs (dict): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key
        solution (dict): Dictionary corresponding to the solution of the JSSP where the key is the machine id, 
        and the value is a list of starting times corresponding to each task within a particular job
        x_max (int, optional): The maximum time to not surpass when drawing. Defaults to None.
    """

    df = []
    if x_max is None:
        x_max = get_result(jobs, solution)
    for job, tasks in solution.items():
        for i, start in enumerate(tasks):
            machine, length = jobs[job][i]
            df.append(dict(Machine=machine,
                           Start=convert_to_datetime(start),
                           Finish=convert_to_datetime(start + length),
                           Job=str(job)))
    
    num_tick_labels = list(range(x_max + 1))
    date_ticks = [convert_to_datetime(x) for x in num_tick_labels]

    fig = px.timeline(df, y="Machine", x_start="Start", x_end="Finish", color="Job")
    fig.update_traces(marker=dict(line=dict(width=3, color='black')), opacity=0.5)
    fig.layout.xaxis.update({
        'tickvals': date_ticks,
        'ticktext': num_tick_labels,
        'range': [convert_to_datetime(0), convert_to_datetime(x_max)]
    })
    fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
    fig.show()


def Q_to_Je(Q,offset=0):
    """ Transform the matrix in QUBO format to a matrix in Ising format

    Args:
        Q (List[List[float]]): The QUBO matrix
        offset (int, optional): The offset to equalize the Ising energt with QUBO energy. Defaults to 0.

    Returns:
        List[List[float]]: The Ising matrix
    """

    num_nodes = len(Q)

    dict_Q = {}
    for i in range(len(Q)):
        for j in range(len(Q[i])):
            if Q[i][j] != 0:
                dict_Q[i, j] = Q[i][j]

    h = {}
    J = {}
    linear_offset = 0.0
    quadratic_offset = 0.0

    J_matrix = np.zeros((num_nodes, num_nodes))
    h_vector = np.zeros(num_nodes)

    for (u, v), bias in dict_Q.items():
        if u == v:
            if u in h:
                h[u] += .5 * bias
            else:
                h[u] = .5 * bias
            linear_offset += bias

        else:
            if bias != 0.0:
                J[(u, v)] = .25 * bias

            if u in h:
                h[u] += .25 * bias
            else:
                h[u] = .25 * bias

            if v in h:
                h[v] += .25 * bias
            else:
                h[v] = .25 * bias
            quadratic_offset += bias

    offset += .5 * linear_offset + .25 * quadratic_offset
    h_dict: dict = h
    J_dict: dict = J

    for i, bias in h_dict.items():
        h_vector[i] = bias

    for (i, j), bias in J_dict.items():
        J_matrix[i][j] = bias

    symmetric_J = make_symmetric(J_matrix)
    Je = reshape_matrix(symmetric_J, h_vector, num_nodes + 1)

    return h_vector,J_matrix,Je,offset


def reshape_matrix(J, h, n):
    """ Creates a new matrix in the Ising model which contains the ferromagnetic 
    field in the first row and first column

    Args:
        J (List[List[float]]): The Ising matrix
        h (List[float]): The ferromagnetic field vector
        n (int): The size of the new matrix

    Returns:
        List[List[float]]: A new matrix containing the J and h
    """
    Je = np.zeros((n, n))
    r, c = 1, 1
    J = np.array(J)
    h = np.array(h)
    Je[r:r + J.shape[0], c:c + J.shape[1]] += J
    Je[1:, 0] = -0.5 * h
    Je[0, 1:] = -0.5 * h
    return Je

def make_symmetric(J):
    """ Create a new symmetric matrix from the J matrix

    Args:
        J (List[List[float]]): The Ising matrix

    Returns:
        List[List[float]]: The matrix J in a symmetric format
    """
    J_sym = 1 / 2 * (J + np.transpose(J))
    return J_sym


def transformToMachineDict(jobs: dict, solution: dict) -> dict:
    """Given a solution to a problem, produces a dictionary indicating the work timeline for each machine.

    Args:
        jobs (dict): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key

        solution (dict): Dictionary corresponding to the solution of the JSSP where the key is the machine id, 
        and the value is a list of starting times corresponding to each task within a particular job

    Returns: 

        dict: {"machine_1": [(job, time_of_operation_start, length), (..., ..., ...), ...],
         "machine_2:: [(..., ..., ...), ...], ...}
    """
    machine_dict = defaultdict(list)
    for key, value in solution.items():
        for i in range(len(value)):
            machine_dict[jobs[key][i][0]].append(
                (key, value[i], jobs[key][i][1]))
    return machine_dict


def checkValidity(jobs: dict, sigma) -> bool:
    """ Checks if given solution fulfills all JSSP constraints.

    Args:
        jobs (dict): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key

        sigma (dict): Dictionary corresponding to the solution of the JSSP where the key is the machine id, 
        and the value is a list of starting times corresponding to each task within a particular job

    Returns:
        bool: true if the solution is valid
    """
    s = sigma[1:]
    
    num_rows = get_num_tasks(jobs)
    num_cols = get_max_time(jobs)
    sigma_matrix = np.reshape(s, (num_rows, num_cols))
    solution = from_sigma_to_solution(sigma_matrix,jobs)
    if solution == 0:
        return False
    # checking if order of operations in jobs is preserved
    for job, operations in jobs.items():
        for i, (operation1, operation2) in enumerate(list(zip(operations[:-1], operations[1:]))):
            if solution[job][i] + operation1[1] > solution[job][i + 1]:
                return False

    machineDict = transformToMachineDict(jobs, solution)

    # checking if no operations using the same machine intersect
    for _, operations in machineDict.items():
        for i, operation1 in enumerate(operations):
            for j, operation2 in enumerate(operations):
                if i == j:
                    continue
                if not (operation1[1] + operation1[2] <= operation2[1] or  # ends before
                        operation2[1] + operation2[2] <= operation1[1]):  # starts after
                    return False
    return True


def create_tasks_from_jobs(jobs: dict):
    """ Convert the jobs dictionary to a list of list

    Args:
        jobs (dict): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key

    Returns:
        List[List[int]]: The list of List corresponding to the same jobs dictionary
    """

    return [task for t in [tasks for _, tasks in jobs.items()] for task in t]


def get_job_index(jobs: dict, operation_index):
    """ Get the job index given the task operation

    Args:
        jobs (dict): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key
        operation_index (int): The index of the operation we wish to have its jobs index

    Returns:
        int: The index of the job containing the corresponding operation
    """

    job_index = 1
    cumulative_num_ops = len(jobs[job_index])
    for _, tasks in jobs.items():
        if operation_index >= cumulative_num_ops:
            job_index += 1
            cumulative_num_ops += len(tasks)
        else:
            return job_index


def get_indexes_tasks_of_job(jobs: dict, job_idx):
    """ Given a job id, returns the starting index and end index 
    corresponding to the operation ids inside this given job

    Args:
        jobs (dict): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key
        job_idx (int): The job id we wish to determine the starting and end index

    Returns:
        tuple: starting operation index and end operation index of the given job
    """

    number_previous_tasks = 0

    for job, tasks in jobs.items():
        if job == job_idx:
            break
        number_previous_tasks += len(tasks)
    job_task_start_index, job_task_end_index = number_previous_tasks, number_previous_tasks + len(jobs[job_idx]) - 1
    return job_task_start_index, job_task_end_index


def make_2D_array(lis):
    """ From a list of list where the lists can have different lengths, produces a new 
    list of lists of equal lengths (remaining elements are set to 0)

    Args:
        lis (List[List[int]]): The input list we wish to make it in matrix format

    Returns:
        List[List[int]]: lis in matrix format
    """
    n = len(lis)
    lengths = np.array([len(x) for x in lis])
    max_len = np.max(lengths)
    arr = np.zeros((n, max_len))

    for i in range(n):
        arr[i, :lengths[i]] = lis[i]
    return arr, lengths
