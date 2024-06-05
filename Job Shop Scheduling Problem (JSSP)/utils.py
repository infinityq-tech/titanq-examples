from collections import defaultdict
from datetime import datetime
import pandas as pd
import plotly.express as px
import numpy as np

# Time scale for the entries of the prod_time_matrix and transfer_time_matrix.
# Value of 1 indicates that prod_time_matrix and transfer_time_matrix are specified in days.
# Value of 7 indicates that prod_time_matrix and transfer_time_matrix are specified in weeks.
# Value of 30 indicates that prod_time_matrix and transfer_time_matrix are specified in months.
time_unit = 1

def read_instance(path: str) -> dict:
    """ Read the JSSP instance from the file path.

    Args:
        path (str): The file path of the JSSP.

    Returns:
        Dict{int:List[Tuple(int,int)]}: The jobs dictionary where the key is the job number
        and the value is a list of execution times of the tasks of the corresponding job key.
    """

    job_dict = defaultdict(list)
    with open(path) as f:
        f.readline()
        for i, line in enumerate(f):
            lint = list(map(int, line.split()))
            job_dict[i] = [x for x in
                               zip(lint[::2],  # machines
                                   lint[1::2]  # operation lengths
                                   )]
        return dict(job_dict)

def get_max_time(jobs):
    """ Compute the maximum time in order to execute all the tasks of the JSSP.

    Args:
        jobs (Dict{int:List[Tuple(int,int)]}): Dictionary where the key is the job number
        and the value is a list of execution times of the tasks of the corresponding job key.

    Returns:
        int: The maximum time to run all the tasks.
    """

    max_time = 0
    for job in jobs.values():
        max_time += sum(a[1] for a in job)
    return max_time

def get_num_machines(jobs: dict):
    """ Compute the number of machines in the JSSP instance.

    Args:
        jobs (Dict{int:List[Tuple(int,int)]}): Dictionary where the key is the job number
        and the value is a list of execution times of the tasks of the corresponding job key.

    Returns:
        int: The number of machines in the JSSP instance.
    """

    return max(max(machine for machine, p in tasks) for _, tasks in jobs.items()) + 1

def get_num_tasks(jobs):
    """ Compute the number of tasks in the JSSP instance.

    Args:
        jobs (Dict{int:List[Tuple(int,int)]}): Dictionary where the key is the job number 
        and the value is a list of execution times of the tasks of the corresponding job key.

    Returns:
        int: The number of tasks in the JSSP instance
    """

    num_tasks = 0

    for _, value in jobs.items():
        num_tasks += len(value)

    return num_tasks

def get_global_index_of_task(jobs, job_id, task_id):
    """ Returns the index of the task in the list of all tasks given the index of the task within the given job.

        Args:
            jobs (Dict{int:List[Tuple(int,int)]}): Dictionary where the key is the job number
                and the value is a list of execution times of the tasks of the corresponding job key.
            job_id (int): The index of the job where the task is in.
            task_id (int): The index of the task within the job.

        Returns:
            int: The index of the given task.
    """

    s = 0
    for job, tasks in jobs.items():
        if job == job_id:
            return s + task_id
        else:
            s += len(tasks)
    return s
       
def extract_solution(start_times, tasks, task_names):
    """Returns the schedule from the state solution.

        Args:
            start_times (np.ndarray[int]): The list containing the start times for each task.
            tasks (dict): The jobs dictionary where the key is the job number and the value is a list of execution times of the tasks of the corresponding job key.
            task_names (List[str]): List containing the tasks names.

        Returns:
            Dict{str:List[Tuples[int,int]]}: The schedule as a dictionary where the keys are the tasks names and the values are the tuples (start_date, end_date) for the corresponding task
    """

    schedule = {}
    for idx, start in enumerate(start_times):
        schedule[task_names[idx]] = (start, start+tasks[idx][1])

    return schedule

def convert_to_datetime(x):
    """Converts an integer into a date.

    Args:
        x (int): Time index.
                 Ex: 15

    Returns:
        Date in YYYY-MM-DD format.
    """

    dt_time = datetime.now().timestamp()
    return datetime.fromtimestamp(x * 24 * 3600 + dt_time).strftime("%Y-%m-%d")

def obtain_group_name(task):
    """Returns the associated job for the given task.

    Args:
        task (str): The task we want to extract the job from.

    Returns:
        str: The associated job for the given task.
            Ex: Job1_Task1 --> Job1
    """
    if task.startswith("Hand"):
        return "Hand-off"
    else:
        return task.split("_")[0]

def plot_schedule(assignments, schedule, available_time_slots, M, unit="weeks", time_unit=1, path="gantt.html"):
    """Plots a GANTT chart given a schedule and assignments.

    Args:
        assignments (Dict{str: str}): Dictionary storing the assignment of tasks to suppliers/machines.
        schedule (Dict{str: Tuple(int, int)}): Dictionary where the keys are tasks, and values are a tuple where the first entry is the start time and the second entry is the end time for the task.
        available_time_slots (np.ndarray[int]): List of available time slots for the machine.
        M (np.ndarray[str]): List of available machines.
                        Ex: ["S1_Casting", "S1_Machining", "S2_Machining", "S3_Casting"]
        unit (str, optional): Unit of time for the schedule; defaults to "weeks".
                              Ex: "weeks"
        time_unit (int): Time scale for the entries of the prod_time_matrix and transfer_time_matrix.
            Value of 1 indicates that prod_time_matrix and transfer_time_matrix are specified in days.
            Value of 7 indicates that prod_time_matrix and transfer_time_matrix are specified in weeks.
            Value of 30 indicates that prod_time_matrix and transfer_time_matrix are specified in months.
        path (str): The HTML file path to save the plot to.

    Returns:
        None
    """

    # Retrieve blocked time slots
    blocked_time_slots = {M[machine]: [(idx*time_unit, idx*time_unit + time_unit) for idx, e in enumerate(
        available_time_slots[machine]) if e == -1] for machine in range(len(M))}

    plot_gantt_chart(blocked_time_slots, schedule, assignments, unit, path)

def plot_gantt_chart(blocked_time_slots, schedule, assignments, unit, path="gantt.html"):
    """Creates a GANTT chart figure for visualization purposes. 

        Args:
            blocked_time_slots (Dict{str: List(Tuple(int, int))}): Dictionary where the keys are the name of suppliers/machines and the values are lists of tuples where the first entry of each tuple is 
            the start date of the blocked times and the second entry is end date of the blocked times. 
                        Ex: {"S1_Casting_M1": [(0, 13), (20, 40)]}
            schedule (Dict{str: Tuple(int, int)}): Dictionary where the keys are tasks, and values are a tuple where the first entry is the start time and the second entry is the end time for the task.
            assignments (Dict{str: str}): Dictionary storing the assignment of tasks to suppliers/machines.
            unit (str, optional): Unit of time for the schedule; defaults to "weeks".
                                Ex: "weeks"
            path (str): HTML file name to save the plot.

        Returns:
            None
    """
    
    color_map = ['#d5d5d5', "#ef553b", "#ff97ff", "#ffa15a", "#abc3f1", "#19d3f3", 
                 "#b6e880", "#fecb52", "#ff6692", "#00cc96", '#cf2918', '#a44155', 
                 '#684502', '#77a540', '#c41cfd', '#d56fa5', '#2ba793', '#826434', 
                 '#cd3098', '#42c1d9', '#2a0514', '#217194', '#7e1dd2', '#b59e8b', 
                 '#62de26', '#ef973f', '#3fd5bc', '#760dda', '#425bbc', '#4f56b8', 
                 '#23247d', '#1dfd0a', '#879345', '#313f19', '#728434', '#21daf1', 
                 '#a0479f', '#021680', '#f0e8ce', '#4b171a']

    # Main data frames containing all necessary data
    df = pd.DataFrame([
        dict(
            Job=obtain_group_name(task),
            Task=task,
            Start=convert_to_datetime(start),
            Finish=convert_to_datetime(end),
            Machine=assignments[task])
        for task, (start, end) in schedule.items()
    ])

    blocked_df = pd.DataFrame([
        dict(
            Job="Blocked",
            Task="Blocked",
            Start=convert_to_datetime(start),
            Finish=convert_to_datetime(end),
            Machine=machine
        )
        for machine in blocked_time_slots.keys() for _, (start, end) in enumerate(blocked_time_slots[machine])
    ])

    # Replicate previous dataframe without Job data. This will be grouped into the first color group since Job is empty.
    grayed_df = pd.DataFrame([
        dict(
            Job="",
            Task=task,
            Start=convert_to_datetime(start),
            Finish=convert_to_datetime(end),
            Machine=assignments[task],
        )
        for task, (start, end) in schedule.items()
    ])
    grayed_blocked_df = pd.DataFrame([
        dict(
            Job="",
            Task="Blocked",
            Start=convert_to_datetime(start),
            Finish=convert_to_datetime(end),
            Machine=machine
        )
        for machine in blocked_time_slots.keys() for _, (start, end) in enumerate(blocked_time_slots[machine])
    ])

    # Combine DataFrames (Grayed blocks are plotted first)
    combined_df = pd.concat([grayed_df, grayed_blocked_df, df, blocked_df])

    # Creating the plot
    fig = px.timeline(
        combined_df,
        x_start="Start",
        x_end="Finish",
        y="Machine",
        color="Job",  # Color is based on part
        hover_name="Task",
        color_discrete_sequence=color_map,
    )

    fig.update_layout(
        title_text='Schedule',
        xaxis_title=f'Time [{unit}]',
        yaxis_title='Machine',
        showlegend=True,
    )

    fig.update_traces(
        selector=dict(name='Blocked'), showlegend=True, marker=dict(color='black', line=dict(width=0.0)),
    )

    num_vals = max_value_schedule(schedule) + 1

    # Split by month
    fig.update_layout(xaxis_range=[convert_to_datetime(
        0), convert_to_datetime(num_vals)])
    fig.update_xaxes(showgrid=True, ticklabelmode="period", dtick="M1")

    # Reverses the ordering of machine along the y-axis
    fig.update_yaxes(autorange="reversed")
    fig.show()
    fig.write_html(path)

def max_value_schedule(schedule):
    """ Returns the end time of the schedule.

    Args:
        schedule (Dict{str: Tuple(int, int)}): Dictionary where the keys are tasks, and values are a tuple where the first entry is the start time and the second entry is the end time for the task.

    Returns:
        int: The end time of the schedule.
    """
    max_value = float('-inf')  # Initialize with negative infinity to handle negative values
    
    for value in schedule.values():
        max_value = max(max_value, value[1])
        
    return max_value

def generate_machine_group(jobs):
    """ Generates a dictionary where the key is the machine, and the values are a list of the tasks assigned to that machine. Task values are generated by get_global_index_of_task.

        Args:
            jobs Dict: Dictionary generated by generate_jobs_dictionary. Keys are job index, keys are job index, values are a list of tasks and production times.

        Returns:
            Dict{int:List[Tuple(int,int)]}: The dictionary where the keys are the machine ids and the values are a list of tuples of (job_id, task_id) assigned to that machine.

    """
    machine_group = dict()
    for job_1, tasks_1 in jobs.items():
        num_tasks = len(tasks_1)
        for task_id in range(num_tasks):
            machine_id_key = tasks_1[task_id][0]

            if not machine_id_key in machine_group:
                machine_group[machine_id_key] = []
            machine_group[machine_id_key].append((job_1, task_id))
    return machine_group

def find_available_time_slots(slots):
    """Returns all the available time slots for a single machine.

    Args:
        slots (np.ndarray[int]): List indicating the availability of the machine.
            Entries of 1 indicate that the machine is available during that time slot.
            Entries of -1 indicate that the machine is not available during that time slot.

    Returns:
        List[Tuple(int, int)]: List of slots where the machine is available. 
            The first entry represents the start time and the second entry represents the end time of the available time slot.
    """
    result_list = []
    current_tuple = None

    tuples = [(idx*time_unit, idx*time_unit + time_unit)
              for idx in np.where(slots == 1)[0]]

    for tpl in tuples:
        if current_tuple is None:
            current_tuple = tpl
        elif current_tuple[1] == tpl[0]:
            current_tuple = (current_tuple[0], tpl[1])
        else:
            result_list.append(
                (current_tuple, current_tuple[1]-current_tuple[0]))
            current_tuple = tpl

    # Append the last tuple
    if current_tuple is not None:
        result_list.append((current_tuple, current_tuple[1]-current_tuple[0]))

    # Sorted available slots in terms of length. If the length of the available slot is smaller that the completion time of the task, then the task won't be scheduled.
    sorted_result = list(
        map(lambda t: t[0], sorted(result_list, key=lambda t: t[1])))

    return sorted_result