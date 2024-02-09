import numpy as np
from utils import *


def generate_qubo(jobs, a, b, c, d, e):
    """ Generate the QUBO matrix of the JSSP problem

    Args:
        jobs (dict): The jobs dictionary where the key is the job number 
        and the value is a list of execution times for the tasks of the corresponding job key
        a (int): The hyperparameter used for incentivizing each operation to be executed exactly once throughout the entire process
        b (int): The hyperparameter used for incentivizing each machine to execute only up to one operation at any given time
        c (int): The hyperparameter used for incentivizing the schedule to follow the order of execution for each operation within a particular job specified from the input file
        d (int): The hyperparameter used for incentivizing machines to perform an operation at the very beginning of the schedule
        e (int): The hyperparameter used for incentivizing the machines to complete all jobs in the shortest possible time

    Returns:
        Numpy.array[float,float]: The QUBO matrix of the JSSP
    """

    tasks_2d = make_2D_array(create_tasks_from_jobs(jobs))
    tasks = tasks_2d[0]

    max_time = get_max_time(jobs)
    num_tasks = get_num_tasks(jobs)
    num_machines = get_num_machines(jobs)
    num_jobs = len(jobs.keys())

    T = max_time
    M = num_machines
    O = num_tasks
    N = num_jobs
    num_nodes = O * max_time

    Q = np.zeros((num_nodes, num_nodes))

    # c*sum(n)sum(k_{n-1}<i<k_n,t+p_i>t_prime)(x_i_t*x_{i+1}_t_prime)
    for n in range(N):
        job_task_start_index, job_task_end_index = get_indexes_tasks_of_job(jobs, n+1)
        for i in range(job_task_start_index, job_task_end_index):
            for t in range(T):
                for t_prime in range(T):
                    if t_prime < t + tasks[i][1]:
                        qrow = i * max_time + t
                        qcol = (i + 1) * max_time + t_prime
                        Q[qrow][qcol] = Q[qrow][qcol] + c

    
    # b*sum(n)sum((i,t,k,t_prime)\in R_m)(x_i_t*x_k_t_prime)
    I = [[i for i in range(O) if tasks[i][0] == m] for m in range(M)]
    A = [[(i, t, k, t_prime) for i in I[m] for t in range(T) for k in I[m] for t_prime in range(T) if i != k and 0 < t_prime - t < tasks[i][1]] for m in range(M)]
    B = [[(i, t, k, t_prime) for i in I[m] for t in range(T) for k in I[m] for t_prime in range(T) if i < k and t == t_prime and tasks[i][1] > 0 and tasks[k][1] > 0] for m in range(M)]

    for m in range(M):
        for i in range(O):
            for k in range(O):
                for t in range(T):
                    for t_prime in range(T):
                        if (i,t,k,t_prime) in A[m] or (i,t,k,t_prime) in B[m]:
                            # print("B done")
                            qrow = i * max_time + t
                            qcol = k * max_time + t_prime
                            Q[qrow][qcol] = Q[qrow][qcol] + b

    # a*sum(i)sum(t)sum(t')(x_i_t*x_i_t_prime)
    for i in range(O):
        for t in range(T):
            for t_prime in range(T):
                qrow = i * max_time + t
                qcol = i * max_time + t_prime
                Q[qrow][qcol] = Q[qrow][qcol] + a


    # -2*a*sum(i)sum(t)(x_i_t)
    for i in range(O):
        for t in range(T):
            qrow = i * max_time + t
            qcol = i * max_time + t
            Q[qrow][qcol] = Q[qrow][qcol] - 2 * a

    # e*sum(i \in L)sum(t \in T) t x_i_t
    L = last_task_for_each_job(jobs)
    for i in L:
        for t in range(T):
            qrow = i * max_time + t
            qcol = i * max_time + t
            Q[qrow][qcol] = Q[qrow][qcol] + e*t
        
    # d*sum(i \in F)sum(t \in T) t x_i_t
    F = first_task_for_each_job(jobs)
    for i in F:
        for t in range(T):
            qrow = i * max_time + t
            qcol = i * max_time + t
            Q[qrow][qcol] = Q[qrow][qcol] + d*t

    Q_sym = make_symmetric(Q)
    
    return Q_sym

