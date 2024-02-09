# Copyright InfinityQ Tech 2024
# Author: Ethan Wang ethan@infintyq.tech

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

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
    
    
def get_stock_corr_matrix(tickers, start, end, period='1YE'):
    '''
    Given a list of stock symbols (tickers), return their correlation matrix (pairwise) over the specified end to start
    date and period of returns. That is, the i,j entry of the returned matrix will be the correlation between
    stocks[i] and stocks[j].
    
    Args:
        tickers (List[str]): List of tickers that we want to generate a correlation matrix for.
        start_date (str): Start date of the time frame we want to sample. Should be in the form "YYYY-MM-DD".
        end_date (str): End date of the time frame we want to sample. Should be in the form "YYYY-MM-DD".
        period (str): Period of returns.
        
    Returns:
        pd.DataFrame[float]: Pandas dataframe with tickers on both axes, filled with pairwise correlation coefficients.
    '''
    daily_adj_closing_prices = yf.download(tickers, start=start, end=end)['Adj Close']
    returns = daily_adj_closing_prices.resample(period).ffill().pct_change()[tickers]
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
    
    
def get_annualized_stock_returns(tickers, start_year, end_year):
    '''
    Given a list of stock symbols (tickers), return their annualized returns from beginning of start_year (January 1)
    to end of end_year (December 31).
    
    Args:
        tickers (List[str]): List of tickers that we want to get the annualized returns for.
        start_year (int): Start year of the time frame we want to sample.
        end_year (int): End year of the time frame we want to sample.
        
    Returns:
        pd.Series[float]: Pandas series of the annualized returns of the given stocks over the specified time period.
    '''
    daily_adj_closing_prices = yf.download(tickers, start=f"{start_year}-01-01", end=f"{end_year}-12-31")['Adj Close']
    annual_returns = daily_adj_closing_prices.resample('1YE').ffill().pct_change()[tickers]
    return ((annual_returns+1).prod() ** (1/(end_year - start_year)) - 1) * 100


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