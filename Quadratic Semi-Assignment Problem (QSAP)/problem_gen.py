# Copyright (c) 2024, InfinityQ Technology Inc.

import numpy as np
import pandas as pd
import os


def generate_bias(production_cost_matrix):
    """ This function flattens the Production Cost Matrix into the bias vector.
    The bias vector represents the vector b in the objective function -1/2 * x^T.W.x - b.x.

    Args:
        production_cost_matrix (np.ndarray[int, int]): The Production Cost Matrix containing information regarding the processing costs associated with a certain material and facility.

    Returns:
        Numpy.array[float32]: The bias vector b to be entered into the TitanQ solver.
    """

    dim_rows = len(production_cost_matrix)
    dim_cols = len(production_cost_matrix[0])
    b = np.zeros(dim_rows*dim_cols, dtype=np.float32)
    for i in range(dim_rows):
        for j in range(dim_cols):
            b[i*dim_cols+j] = production_cost_matrix[i][j]
    return b


def generate_weights(flow_matrix, distance_matrix):
    """ Generates the weight matrix based on the Flow Matrix and Distance Matrix.
    The weight matrix represents the matrix W in the objective function -1/2 * x^T.W.x - b.x.

    Args:
        flow_matrix (np.ndarray[int, int]): The Flow Matrix describing the dependencies between the processing of materials.
        distance_matrix (np.ndarray[int, int]): The distance matrix containing information regarding the distance between locations.

    Returns:
        Numpy.array[float32, float32]: The weight matrix to be entered into the TitanQ solver.
    """

    n = len(flow_matrix)
    m = len(distance_matrix)
    dim = n*m
    W = np.zeros((dim, dim), dtype=np.float32)

    for i in range(dim):
        for j in range(dim):
            W[i][j] = flow_matrix[int(i/m)][int(j/m)] * distance_matrix[int(i % m)][int(j % m)]

    return W

def generate_assignment(solution_matrix, material_names, facility_names):
    """ Generates the assignment of materials to locations and facilities given a solution to the QSAP.

    Args:
        solution_matrix (np.ndarray[int, int]): The result vector returned by TitanQ reshaped into a 2d array.
        material_names (List[str]): The list of material names in the format "Material_#".
        facility_names (List[str]): The list of facility names in the format "Facility_#".

    Returns:
        Dict[str, str]: The QSAP assignment of material to locations/facilities where key = material name and value = location/facility name.
    """

    dim_rows = len(solution_matrix)
    dim_cols = len(solution_matrix[0])
    assignment = {}
    for i in range(dim_rows):
        for j in range(dim_cols):
            if solution_matrix[i][j] == 1:
                assignment[material_names[i]] = facility_names[j]
    return assignment


def reshape_result_vector(result_vector, num_materials, num_facilities):
    """ Transforms the result vector returned by TitanQ into a 2d array.

    Args:
        result_vector (np.ndarray[int]): The result vector returned by TitanQ reshaped into a 2d array.
        num_materials (int): The number of materials to be completed.
        num_facilities (int): The number of facilities at all locations.

    Returns:
        Numpy.array[float32, float32]: The result vector as a 2d array.
    """

    result_matrix = np.zeros((num_materials, num_facilities), dtype=int)

    for i in range(num_materials):
        for j in range(num_facilities):
            result_matrix[i][j] = result_vector[i*num_facilities + j]

    return result_matrix

def save_assignment(result_matrix, material_names, facility_names):
    """ Save the assignment in an Excel file.

    Args:
        result_matrix (np.ndarray[int][int]): The result matrix returned by TitanQ reshaped into a 2d array.
        material_names (List[str]): The list of task names in the format "Component_Step#".
        facility_names (List[str]): The list of facility names in the format "Vendor_Process".

    Returns:
          None
    """

    # Transform solution matrix into Pandas DataFrame where the rows are the materials and the columns are the facilities
    df = pd.DataFrame(data=result_matrix,
                      index=material_names,
                      columns=facility_names)
    
    folder_path = "outputs"
    file_name = 'solution.xlsx'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Combine folder path and file name
    file_path = os.path.join(folder_path, file_name)

    # Check if the file exists. Create the file if it does not exist.
    if not os.path.exists(file_path):
        # Save the DataFrame as Excel file
        df.to_excel(file_path)