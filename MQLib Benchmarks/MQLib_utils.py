import numpy as np
import csv

def read_instance(path):
    """
    Read in an instance file and return the corresponding QUBO weight matrix.

    Args:
        path (str): File path of the instance to load data from.

    Returns:
        nparray[nparray[float]]: Initial weight matrix for the TitanQ solver constructed from the instance file.
    """
    with open(path, 'r') as file:
        # Skip first line
        next(file)
        
        # Diagonal entries
        diagonal_entries = []

        # True if we are currently still reading in the diagonal entries, false otherwise
        diagonal_flag = True
        
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue
            
            line_data = line.strip().split()
            
            # Check if this line is a diagonal entry
            if diagonal_flag:
                if line_data[0] == line_data[1]:
                    diagonal_entries.append(float(line_data[2]))
                else:
                    size = len(diagonal_entries)
                    weights_matrix = np.zeros((size, size), dtype=np.float64)
                    np.fill_diagonal(weights_matrix, np.array(diagonal_entries, dtype=np.float64))
                    diagonal_flag = False
            
            # Run if we are finished reading in diagonal entries
            if not diagonal_flag:
                weights_matrix[int(line_data[0]), int(line_data[1])] = np.float64(line_data[2])
        
    return weights_matrix


def generate_weights_bias(instance, instance_file_path):
    """
    Returns weights matrix and bias vector given instance.
    
    Args:
        instance (str): Name of the instance to generate updated weights matrix and bias vector from.
        instance_file_path (str): File path where the instance file is stored.

    Returns:
        nparray[nparray[float]]: Updated weight matrix for the TitanQ solver constructed from the instance file.
        nparray[float]: Bias vector for the TitanQ solver constructed from the instance file.
    """
    # Read in QUBO matrix from file without altering
    QUBO_matrix = read_instance(instance_file_path + "/" + instance).astype(np.float32)
    
    # Bias vector
    bias_vector = np.array(QUBO_matrix, dtype=np.float32).diagonal()
    
    # Weights matrix
    weights_matrix = QUBO_matrix + np.transpose(QUBO_matrix)
    np.fill_diagonal(weights_matrix, 0)
    
    return weights_matrix, bias_vector


def load_hyperparameters(instance_name):
    """
    Returns optimal hyperparameters for the TitanQ solver for a specified MQLib instance. 

    Args:
        instance_name (str): Name of the MQLib instance to load associated hyperparameters from

    Returns:
        float: Minimum temperature hyperparameter used for setting up beta values
        float: Maximum temperature hyperparameter used for setting up beta values
        float: Coupling multiplier hyperparameter used for configuring TitanQ solver
        int:   Number of chains hyperparameter for configuring TitanQ solver 
        int:   Number of engines hyperparameter for configuring TitanQ solver 
    """
    with open('instances_best_hyperparameters.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader, None) #Skip header row

        hyperparameters_dict = {}
        for row in reader:
            hyperparameters_dict[row[0]] = row[1:]

    T_min = float(hyperparameters_dict[instance_name][0])
    T_max = float(hyperparameters_dict[instance_name][1])
    coupling_mult = float(hyperparameters_dict[instance_name][2])
    num_chains = int(hyperparameters_dict[instance_name][3])
    num_engines = int(hyperparameters_dict[instance_name][4])

    return T_min, T_max, coupling_mult, num_chains, num_engines
