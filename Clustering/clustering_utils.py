# Copyright InfinityQ Tech 2024
# Author: Brian Mao brian@infinityq.tech
# Date: Nov 29, 2024

import csv
import numpy as np
from math import sqrt

def gen_dist_matrix(coords:list[list[float]]) -> np.ndarray[float]:
    """
    Generates the associated distance matrix from a set of coordinates.

    Args:
        coords (list[list[float]]): List of coordinates to be clustered.
    Returns:
        nparray[float]: Associated distance matrix where entry [i,j] contains the distance of point i to point j.
    """
    norm_coords = np.array(coords)
    norm_coords[:, 0] -= np.min(norm_coords[:, 0])
    norm_coords[:, 1] -= np.min(norm_coords[:, 1])
    norm_coords[:, 0] = norm_coords[:, 0]/np.max(norm_coords[:, 0])
    norm_coords[:, 1] = norm_coords[:, 1]/np.max(norm_coords[:, 1])

    dist_mat = np.zeros((len(norm_coords), len(norm_coords)))
    for i in range(len(norm_coords)):
        for j in range(len(norm_coords)):
            x0 = norm_coords[i]
            x1 = norm_coords[j]
            dist_mat[i, j] = np.sqrt((x0[0] - x1[0])**2 + (x0[1] - x1[1])**2)
    return dist_mat


def gen_Jh_cluster(dist_mat:np.ndarray[float], coords:list[list[float]], n_clusters:int, lambda_scaling_factor:float, k_avg:float, B:int) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Generates the J-matrix and h-vector for a clustering problem while incorporating Lagrangian term for cluster rebalancing.

    Args:
        dist_mat (nparray[float]): Distance matrix for coordinates where entry [i,j] contains the distance of point i to point j.
        coords (list[list[float]]): List of coordinates to be clustered.
        n_clusters (int): Number of clusters to be generated by the overall RBM clustering algorithm.
        lambda_scaling_factor (float): Hyperparameter to tune the effect of the Lagrangian term for cluster rebalancing.
        k_avg (float): Target number of coordinates to include within each cluster on average (Often set to ncoords/nclusters).
        B (int): Hyperparameter that gets multiplied to entries from the distance matrix.

    Returns:
        nparray[float]: Associated J-matrix of size (n_coords*n_clusters) x (n_coords*n_clusters) for the particular clustering problem.
        nparray[float]: Associated h-vector of length n_coords*n_clusters for the particular clustering problem.
    """
    n_coords = len(coords)
    J = np.zeros((n_coords*n_clusters, n_coords*n_clusters), dtype = np.float32)
    h = np.zeros((n_coords*n_clusters), dtype = np.float32)

    #Incorporating distances into J-matrix
    for i, row in enumerate(dist_mat):
        for j, val in enumerate(row):
            for k in range(n_clusters):
                J[i*n_clusters + k][j*n_clusters + k] += B*val
                J[j*n_clusters+k][i*n_clusters+k] += B*val

    #Cluster Rebalancing
    for i in range(n_coords*n_clusters):
        h[i] += -2*k_avg*lambda_scaling_factor

    for i, row in enumerate(dist_mat):
        for j, val in enumerate(row):
            for k in range(n_clusters):
                J[i*n_clusters + k][j*n_clusters + k] += lambda_scaling_factor

    return J, h


def load_hyperparameters(n_coords:int, n_clusters:int, lambda_scaling_factor:float) -> tuple[float, float, float, int, int] | None: 
    """
    Returns optimal hyperparameters for the TitanQ solver for specified instances. 

    Args:
        n_coords (int): Total number of points to cluster
        n_clusters (int): Number of clusters to extract from the data set
        lambda_scaling_factor (float): Hyperparameter to tune the effect of the Lagrangian term for cluster rebalancing.

    Returns:
        float: Minimum temperature hyperparameter used for setting up beta values
        float: Maximum temperature hyperparameter used for setting up beta values
        float: Coupling multiplier hyperparameter used for configuring TitanQ solver
        int:   Number of chains hyperparameter for configuring TitanQ solver 
        int:   Number of engines hyperparameter for configuring TitanQ solver 
    """
    with open('clustering_hyperparameters.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader, None) #Skip header row

        hyperparameters_dict = {}
        for row in reader:
            hyperparameters_dict[(int(row[0]), int(row[1]), float(row[2]))] = row[3:]

    if (n_coords, n_clusters, lambda_scaling_factor) in hyperparameters_dict:
        T_min = float(hyperparameters_dict[(n_coords, n_clusters, lambda_scaling_factor)][0])
        T_max = float(hyperparameters_dict[(n_coords, n_clusters, lambda_scaling_factor)][1])
        coupling_mult = float(hyperparameters_dict[(n_coords, n_clusters, lambda_scaling_factor)][2])
        num_chains = int(hyperparameters_dict[(n_coords, n_clusters, lambda_scaling_factor)][3])
        num_engines = int(hyperparameters_dict[(n_coords, n_clusters, lambda_scaling_factor)][4])
        timeout = int(hyperparameters_dict[(n_coords, n_clusters, lambda_scaling_factor)][5])
        return T_min, T_max, coupling_mult, num_chains, num_engines, timeout
    else:
        return None


def calculate_centroid(coordinates:list[tuple[float,float]]) -> tuple[float, float]:
    """
    Calculates and returns the centroid among a list of coordinates.

    Args:
        coordinates (list[tuple(float,float)]): List of coordinates to find the centroid among.

    Returns:
        tuple(float, float): The centroid among a list of coordinates.
    """
    sum_x, sum_y = 0, 0
    for coord in coordinates:
        sum_x += coord[0]
        sum_y += coord[1]
    return (sum_x/len(coordinates), sum_y/len(coordinates))


def centroid_diameter_distance(cluster:list[tuple[float,float]]) -> float:
    """
    Calculates and returns the centroid diameter distance, also reffered to as the intracluster distance for a particular cluster.

    Args:
        cluster (list[tuple(float,float)]): List of coordinates within a particular cluster.

    Returns:
        float: The centroid diameter distance (intracluster distance).
    """
    centroid = calculate_centroid(cluster)
    total_distance = 0
    for i in range(len(cluster)):
        total_distance += sqrt((cluster[i][0]-centroid[0])**2 + (cluster[i][1]-centroid[1])**2) 
    return 2*(total_distance/len(cluster))


def intracluster_distance_calculation(overall_clusters:list[list[tuple[float,float]]]) -> list[float]:
    """
    Generates a list of intracluster distances across all clusters.

    Args:
        overall_clusters (list[list[tuple(float,float)]]): List of clusters where each cluster is a list of coordinates.

    Returns:
        list[float]: List containing intracluster distances among all clusters.
    """
    intracluster_distances = []
    for i in overall_clusters:
        intracluster_distances.append(centroid_diameter_distance(overall_clusters[i]))
    return intracluster_distances


def intercluster_distance_calculation(overall_clusters:list[list[tuple[float,float]]]) -> list[float]:
    """
    Generates a list of intercluster distances between each pair of clusters.

    Args:
        overall_clusters (list[list[tuple(float,float)]]): List of clusters where each cluster is a list of coordinates.

    Returns:
        list[float]: List containing intercluster distances among all clusters.
    """
    if len(overall_clusters) == 1:
        return [0]

    intercluster_distances = []
    for i in overall_clusters: 
        for j in overall_clusters:
            if i < j: 
                centroid_1 = calculate_centroid(overall_clusters[i])
                centroid_2 = calculate_centroid(overall_clusters[j])
                intercluster_distances.append(sqrt((centroid_1[0]-centroid_2[0])**2 + (centroid_1[1]-centroid_2[1])**2))
    return intercluster_distances
