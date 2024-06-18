# Copyright (c) 2024, InfinityQ Technology Inc.

import numpy as np

def read_instance(path):
    """
    Read in list of tickers and weights (if given) from the file path.

    Args:
        path (str): The relative file path of the list of tickers.

    Returns:
        List[str]: List of tickers displayed row by row in the file.
        nparray[float]: List of weights of each ticker. This return value is None if no weights are provided.
    """
    with open(path) as f:
        lines = [line.rstrip().split() for line in f]
        tickers = [line[0] for line in lines]
        
        # Check if weights are given
        if len(lines[0]) == 2:
            weights = np.array([line[1] for line in lines]).astype('float32')
        else:
            weights = None
    
    return tickers, weights
    
    
def get_stock_daily_returns(prices):
    '''
    Given a dataframe of daily stock prices, return their (daily) returns.
    
    Args:
        prices (pd.DataFrame[float]): Time indexed dataframe of daily stock prices.
        
    Returns:
        pd.DataFrame[float]: Pandas dataframe with daily stock returns with the same time index as the input.
    '''
    return prices.ffill().pct_change()


def daily_to_periodic_returns(daily_returns, period):
    '''
    Converts a dataframe of daily returns to periodic returns (e.g. monthly, annually).
    
    Args:
        daily_returns (pd.DataFrame[float]): Time indexed dataframe of daily stock returns.
        
    Returns:
        pd.DataFrame[float]: Pandas dataframe of periodic stock returns.
    '''
    return daily_returns.resample(period).agg(lambda x: (x + 1).prod() - 1)

    
def get_stock_corr_matrix(returns):
    '''
    Given a dataframe of stock returns, return their correlation matrix (pairwise).
    
    Args:
        returns (pd.DataFrame[float]): Time indexed dataframe of stock returns.
        
    Returns:
        pd.DataFrame[float]: Pandas dataframe with tickers on both axes, filled with pairwise correlation coefficients.
    '''
    return returns.corr()
  
    
def corr_to_J_matrix(corr_matrix, theta):
    '''
    Generate the adjacency (J) matrix of a market correlation graph given the correlation matrix of the stocks and the 
    parameter theta.
    
    Args:
        corr_matrix (pd.DataFrame[float]): Correlation matrix of stocks.
        theta (float): Theta parameter. Threshold used for determining what is considered correlated vs uncorrelated.
                       Value should be in the closed interval [0, 1].
                       
    Returns:
        nparray[nparray[float]]: Adjacency matrix corresponding to the given correlation matrix and theta parameter.
    '''
    J_matrix = corr_matrix.to_numpy()
    np.fill_diagonal(J_matrix, 0)
    return (abs(J_matrix) >= theta).astype('float32')


def get_stock_mean_returns(returns):
    '''
    Given a dataframe of stock returns, return the (geometric) mean return of each stock (column).
    
    Args:
        returns (pd.DataFrame[float]): Time indexed dataframe of stock returns.
        
    Returns:
        pd.DataFrame[float]: Pandas dataframe with the mean return of each stock.
    '''
    return (np.exp(np.log(returns + 1).mean()) - 1) * 100


def get_stock_stds(returns):
    '''
    Given a dataframe of stock returns, return the standard deviation of each stock's returns.
    
    Args:
        returns (pd.DataFrame[float]): Time indexed dataframe of stock returns.
        
    Returns:
        pd.DataFrame[float]: Pandas dataframe with the standard deviation in returns of each stock.
    '''
    return (returns * 100).std()


def verify_independent_set(J_matrix, vertices, labels):
    '''
    Given an adjacency matrix (J_matrix) and a set of vertices, return all pairs of vertices in the set that are
    connected (thus violating independence).
    
    Args:
        J_matrix (nparray[nparray[float]]): Adjacency matrix of the graph.
        vertices (nparray[float]): Binary vector indicating what subset of vertices we are checking independence for.
        labels (List[str]): Labels for the vertices.
        
    Returns:
        List[[str, str]]: Pairs of labelled vertices in the given set that are connected.
    '''
    violations = []
    for i in range(J_matrix.shape[0]):
        for j in range(i):
            if J_matrix[i][j] == 1 and vertices[i] == 1 and vertices[j] == 1:
                violations.append([labels[i], labels[j]])
    
    return violations