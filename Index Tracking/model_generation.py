import yfinance as yf
import pandas as pd
import numpy as np
from gurobipy import GRB
from typing import Tuple, Any 
from docplex.mp.model import Model

data_dir = "market_data/"
#Load S&P 500 symbols from a text file
sp500_symbols = np.loadtxt(data_dir + 'sp500_symbols.txt', dtype=str).tolist()


def load_cache(year: str = "2021", ind_symbol: str = "^GSPC") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and process historical adjusted closing prices and daily returns for S&P 500 stocks and the benchmark index from CSV files for a given year.

    Parameters:
    year (str): The year for which the data is to be loaded. Default is "2021".
    ind_symbol (str): The ticker symbol of the benchmark index. Default is "^GSPC" for S&P 500.

    Returns:
    tuple: A tuple containing four pandas objects:
           - stock_data (pd.DataFrame): Historical adjusted closing prices for S&P 500 stocks.
           - stock_returns (pd.DataFrame): Daily returns for S&P 500 stocks.
           - index_data (pd.DataFrame): Historical adjusted closing prices for the benchmark index.
           - index_returns (pd.Series): Daily returns for the benchmark index.
    """
    stock_data = pd.read_csv(data_dir + f'stock_price_{year}.csv', index_col=0, parse_dates=True)
    index_data = pd.read_csv(data_dir + f'index_price_{year}.csv', index_col=0, parse_dates=True)

    #Cleaning and aligning dates in stock and index data
    ind_intersect: pd.DatetimeIndex = index_data.index.intersection(stock_data.index)
    index_data = index_data.loc[ind_intersect].ffill()
    stock_data = stock_data.loc[ind_intersect].ffill()

    stock_returns = stock_data.pct_change().fillna(value=0)

    #Pull the data for the specified index
    index_data = index_data[ind_symbol]
    index_returns = index_data.pct_change().fillna(value=0)

    return stock_data, stock_returns, index_data, index_returns


def download_data(index_name: str = "^GSPC",
                   start_date: str = "2021-01-01", 
                   end_date: str = "2022-01-01") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Download historical adjusted closing prices for S&P 500 stocks and a specified index.
    Calculate daily returns for both the stocks and the index.

    Parameters:
    index_name (str): The ticker symbol of the index. Default is "^GSPC" for S&P 500
    start_date (str): The start date for downloading data in "YYYY-MM-DD" format. Default is "2021-01-01"
    end_date (str): The end date for downloading data in "YYYY-MM-DD" format. Default is "2022-01-01"

    Returns:
    tuple: A tuple containing four pandas Series:
           - stock_data: Historical adjusted closing prices for S&P 500 stocks.
           - index_data: Historical adjusted closing prices for the specified index.
           - stock_returns: Daily returns for S&P 500 stocks.
           - index_returns: Daily returns for the specified index.
    """

    stock_data: pd.DataFrame = yf.download(sp500_symbols, start=start_date, end=end_date, timeout=60, ignore_tz=False)['Adj Close']
    index_data: pd.Series = yf.download(index_name, start=start_date, end=end_date, timeout=60, ignore_tz=False)['Adj Close']
    stock_data.index = stock_data.index.tz_convert("UTC").normalize()
    index_data.index = index_data.index.tz_convert("UTC").normalize()

    ind_intersect: pd.DatetimeIndex = index_data.index.intersection(stock_data.index)
    index_data = index_data.loc[ind_intersect].ffill()
    stock_data = stock_data.loc[ind_intersect].ffill()

    index_returns: pd.Series = index_data.pct_change().fillna(value=0)
    stock_returns: pd.DataFrame = stock_data.pct_change().fillna(value=0)

    return stock_data, stock_returns, index_data, index_returns


def get_objective(stock_returns: pd.DataFrame, 
                  index_returns: pd.Series, 
                  max_invest: int = 16) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate the objective function for a index tracking optimization problem. The objective function is a quadratic form with a linear term and a constant term.

    Parameters:
    stock_returns (pandas.DataFrame): A DataFrame containing historical stock returns. Each column represents a stock, and each row represents a time period.
    index_returns (pandas.Series): A Series containing historical index returns. Each value represents a time period.
    max_invest (int): The maximum investment amount per stock. Default is 16.

    Returns:
    tuple: A tuple containing three numpy arrays:
        - W (numpy.ndarray): A 2D array representing the quadratic term of the objective function.
        - b (numpy.ndarray): A 1D array representing the linear term of the objective function.
        - offset (float): A scalar representing the constant term of the objective function.
    """
    # Convert dataframes to numpy arrays for optimization
    r_it: np.ndarray = stock_returns.to_numpy().T  # Stock returns (N x T)
    R_t: np.ndarray = index_returns.to_numpy()   # Index returns (T)
    # Number of stocks N and number of time periods 
    N: int
    T: int
    N, T = r_it.shape
    W: np.ndarray = np.zeros((N, N), dtype=np.float32)
    b: np.ndarray = np.zeros(N, dtype=np.float32)
    offset: float = 0.0

    for t in range(T):
        W += 2 * np.outer(r_it[:, t]/max_invest, r_it[:, t]/max_invest)
        b += -2 * r_it[:, t]/max_invest * R_t[t]
        offset += R_t[t] * R_t[t]

    return W, b, offset


def calc_returns(stock_returns: pd.DataFrame, 
                 solution: np.ndarray) -> np.ndarray:
    """
    Calculate the portfolio returns based on stock returns and investment solution.

    This function computes the returns of a portfolio given the returns of individual stocks
    and the investment allocation solution.

    Parameters:
    stock_returns : pd.DataFrame
        A DataFrame containing historical stock returns.
        Each column represents a stock, and each row represents a time period.
    solution : np.ndarray
        An array representing the investment allocation for each stock.

    Returns:
    np.ndarray
        An array of portfolio returns for each time period.

    Notes:
    The function assumes that the global variable `sp500_symbols` is available and
    contains the stock symbols in the same order as the columns in `stock_returns`.
    """
    N = len(stock_returns.columns)
    portfolio_returns: np.ndarray = np.zeros(len(stock_returns.index))
    total_investment: float = sum(solution)
    for i, time_ind in enumerate(stock_returns.index):
        for j in range(N):
            stock_tick: str = sp500_symbols[j]
            portfolio_returns[i] += solution[j] * stock_returns.loc[time_ind, stock_tick] / total_investment
    return portfolio_returns


def analyze_results_gurobi(
    model: Any,
    x: Any,
    stock_init_price: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    offset: float
) -> Tuple[np.ndarray, float]:
    """
    Analyze the results from the Gurobi optimization model for the index tracking problem.

    This function processes the solution provided by the Gurobi solver,
    calculates the objective function value, and prints various metrics about the solution.

    Parameters:
    model (Any): The Gurobi model object after optimization.
    x (Any): The decision variables from the Gurobi model.
    stock_init_price (np.ndarray): Initial prices of the stocks.
    W (np.ndarray): Quadratic term of the objective function.
    b (np.ndarray): Linear term of the objective function.
    offset (float): Constant term of the objective function.

    Returns:
    Tuple[np.ndarray, float]: A tuple containing:
        - best_solution (np.ndarray): The optimal solution vector.
        - best_obj (float): The objective function value for the optimal solution.

    Prints:
    - Optimization status (optimal or not)
    - Total tracking error
    - Budget used
    - Number of individual assets used
    - Portfolio stock tickers and weights
    """
    N = len(b)
    # Display results
    if model.status == GRB.OPTIMAL:
        print("Found optimal weights!")
    else:
        print("No optimal solution found. Using generated weights")
    # Optimal weights
    optimal_weights = [x[i].X for i in range(N)]

    best_solution = np.array(optimal_weights)

    N = len(stock_init_price)

    print("==================================================================")
    best_obj = 0.5 * best_solution @ W @ best_solution + best_solution @ b + offset
    print("Total Tracking Error sum(index_return - portfolio_return)^2:", best_obj)
    print("Budget Used:$", sum(stock_init_price[i] * best_solution[i] for i in range(N)))
    print("Individual Assets Used:", sum([x > 0.5 for x in best_solution]))
    stock_tickers: str = ""
    weights: str = ""
    for i in range(len(best_solution)):
        if best_solution[i] > 0.5:
            stock_tickers += f"{sp500_symbols[i]}, "
            weights += f"{best_solution[i]}, "
    print("Portfolio Stock Tickers:", stock_tickers)
    print("Portfolio weights:", weights)
    print("==================================================================")

    return best_solution, best_obj

def analyze_results_cplex(
    model: Any,
    x: Any,
    stock_init_price: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    offset: float
) -> Tuple[np.ndarray, float]:
    """
    Analyze the results from the CPLEX optimization model for the index tracking problem.

    This function processes the solution provided by the CPLEX solver,
    calculates the objective function value, and prints various metrics about the solution.

    Parameters:
    model (Any): The CPLEX model object after optimization.
    x (Any): The decision variables from the CPLEX model.
    stock_init_price (np.ndarray): Initial prices of the stocks.
    W (np.ndarray): Quadratic term of the objective function.
    b (np.ndarray): Linear term of the objective function.
    offset (float): Constant term of the objective function.

    Returns:
    Tuple[np.ndarray, float]: A tuple containing:
        - best_solution (np.ndarray): The optimal solution vector.
        - best_obj (float): The objective function value for the optimal solution.

    Prints:
    - Optimization status (optimal or not)
    - Total tracking error
    - Budget used
    - Number of individual assets used
    - Portfolio stock tickers and weights
    """
    N = len(b)
    # Display results
    if model.solve_details.status == 'optimal':
        print("Found optimal weights!")
    else:
        print("No optimal solution found. Using generated weights")
        return None, None
    
    # Optimal weights    
    optimal_weights = [var.solution_value for var in x]

    best_solution = np.array(optimal_weights)

    N = len(stock_init_price)

    print("==================================================================")
    best_obj = 0.5 * best_solution @ W @ best_solution + best_solution @ b + offset
    print("Total Tracking Error sum(index_return - portfolio_return)^2:", best_obj)
    print("Budget Used:$", sum(stock_init_price[i] * best_solution[i] for i in range(N)))
    print("Individual Assets Used:", sum([x > 0.5 for x in best_solution]))
    stock_tickers: str = ""
    weights: str = ""
    for i in range(len(best_solution)):
        if best_solution[i] > 0.5:
            stock_tickers += f"{sp500_symbols[i]}, "
            weights += f"{best_solution[i]}, "
    print("Portfolio Stock Tickers:", stock_tickers)
    print("Portfolio weights:", weights)
    print("==================================================================")

    return best_solution, best_obj


def analyze_results_titanq(
    response: Any,
    stock_init_price: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    offset: float
) -> Tuple[np.ndarray, float]:
    """
    Analyze the results from TitanQ solver for the index tracking problem.

    This function processes the solutions provided by the TitanQ solver,
    calculates the objective function value for each solution, and identifies
    the best solution that satisfies the constraints.

    Parameters:
    response (Any): The response object from TitanQ solver containing the results.
    stock_init_price (np.ndarray): Initial prices of the stocks.
    sp500_symbols (List[str]): List of S&P 500 stock symbols.
    W (np.ndarray): Quadratic term of the objective function.
    b (np.ndarray): Linear term of the objective function.
    offset (float): Constant term of the objective function.

    Returns:
    Tuple[np.ndarray, float]: A tuple containing:
        - best_solution (np.ndarray): The best solution vector.
        - best_obj (float): The objective function value for the best solution.

    Prints:
    - Analysis for each solution including:
        - Solution number
        - Total tracking error
        - Budget used
        - Number of individual assets used
        - Portfolio stock tickers and weights
    - Constraint violations for all solutions
    """
    print("Analyzing Results")
    best_solution: np.ndarray = response.result_items()[0][1]
    best_obj: float = 1e8
    N: int = len(stock_init_price)
    
    for solution_ind, solution in enumerate(response.result_items()):
        print("==================================================================")
        print(f"Solution #{solution_ind}")
        obj: float = 0.5 * solution[1] @ W @ solution[1] + solution[1] @ b + offset
        print("Total Tracking Error sum(index_return - portfolio_return)^2:", obj)
        print("Budget Used:$", sum(stock_init_price[i] * solution[1][i] for i in range(N)))
        print("Individual Assets Used:", sum([x > 0.5 for x in solution[1]]))
        stock_tickers: str = ""
        weights: str = ""
        for i in range(len(solution[1])):
            if solution[1][i] > 0.5:
                stock_tickers += f"{sp500_symbols[i]}, "
                weights += f"{solution[1][i]}, "
        print("Portfolio Stock Tickers:", stock_tickers)
        print("Portfolio weights:", weights)
        print("Constraint Violations:", response.constraint_violations()[0][solution_ind])
        print("==================================================================")
        if obj < best_obj and (response.constraint_violations()[0][solution_ind] == 0):
            best_solution = solution[1]
            best_obj = obj

    return best_solution, best_obj


Tmin: float = 1e-3
Tmax: float = 10
num_chains: int = 32
num_engines: int = 4
timeout: int = 3
beta: np.ndarray = 1/np.geomspace(Tmin, Tmax, num=num_chains)

titanq_params: dict = {
    "num_chains": num_chains,
    "num_engines": num_engines,
    "timeout_in_secs": timeout,
    "beta": beta
}