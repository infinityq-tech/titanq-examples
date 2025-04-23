# Copyright (c) 2025, InfinityQ Technology Inc.

import numpy as np
import networkx as nx
import pylab
from collections import defaultdict
import pandas as pd
import os
import requests
from requests.exceptions import SSLError
import json
from titanq import Model, Vtype, Target
from contextlib import redirect_stdout

def generate_data():
    """
    Loads minute aggregated forex data from January 2, 2025 into csv file.
    Returns:
        None
    Loads:
        csv files of forex data in directory 'instances/'.
    """
    # Loading in file from public s3 bucket
    file_name = "2025-01-02.csv.gz"
    url = f"https://s3.us-east-1.amazonaws.com/titanq.example/currency-arbitrage-example/{file_name}"

    try:
        # First attempt with SSL verification
        response = requests.get(url)
        response.raise_for_status()

    except SSLError as ssl_error:
        print("SSL verification failed, retrying without verification...")
        # Retry without SSL verification
        response = requests.get(url, verify=False)
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")
        raise

    # Save the file
    with open(f"./instances/{file_name}", 'wb') as f:
        f.write(response.content)
    print(f"Downloaded to '{file_name}'")

    response = requests.get(url)
    response.raise_for_status()

    with open(f"./instances/{file_name}", 'wb') as f:
        f.write(response.content)
    
def load_data(full_dataset, instance):
    """
    Read in bool to use full dataset and instance, and either loads minute aggregated forex data into
    csv files or reads chosen instance file.

    Args:
        full_dataset (bool): A bool indicating whether the full dataset from January 2, 2025 was loaded or not.
        instance (str): Example of choice if user chooses not to load full dataset.

    Returns:
        None, None if no valid date entered.
        List[str]: List of currencies.
        pd.DataFrame: Pandas dataframe containing exchange rate data.

    Loads:
        csv file of forex data for January 2, 2025 in folder 'instances/'.
    """
    currencies = []
    if full_dataset:
        try:
            rates = pd.read_csv(f"./instances/2025-01-02.csv.gz")

        except:
            print("No data found.")
            return None, None
        
        # Store all exchange pairs
        rates = rates.drop_duplicates(subset="ticker", keep="last")

        currency_pairs = rates["ticker"].values
        print(f"Number of active currency pairs: {len(currency_pairs)}")
        
        # Store all currencies
        for pair in currency_pairs:
            base, quote = pair[2:].split("-")
            currencies.append(base)
            currencies.append(quote)

        currencies = sorted(np.unique(currencies))
        df = pd.DataFrame(index=currencies, columns=currencies)

        # Populating exchange rate dataframe
        for _, row in rates.iterrows():
            base, quote = row["ticker"][2:].split("-")
            df.at[base, quote] = row["close"]

    else:
        try:
            with open(f"instances/symbols/{instance}") as f:
                lines = [line.rstrip().split() for line in f]
                currencies = [line[0] for line in lines]
        except:
            print("Please set your instance to either currencies_6 or currencies_25.")
            return

        # Loading exchange rates from file
        with open(f"instances/exch_rates/{instance}.json", "r") as f:
            exchange_rates = json.load(f)

        # Converting rates into pandas dataframe
        df = pd.DataFrame(index=currencies, columns=currencies)
        for pair, rate in exchange_rates.items():
            base, quote = pair[2:].split("-")
            df.at[base, quote] = rate

    df = df.where(pd.notna(df), 0.)

    return currencies, df

def find_exchange_cycles(exchanges):
    '''
    Reads in list of exchanges made in format "XXX/YYY" where "XXX" is the first currency and "YYY" is the second,
    and outputs a dictionary where each key is a cycle that maps to the profit made in that cycle. 
    Prints the different cycles.

    Args:
        exchanges (nparray[str]): List of exchanges made.

    Returns:
        list[list[str]]: List of cycles.
    '''
    if not exchanges:
        return exchanges
    
    graph = defaultdict(list)
    
    # Build graph
    for pair in exchanges:
        a, b = pair.split('/')
        graph[a].append(b)

    cycles = set()  # Use a set to prevent duplicates

    def dfs(node, start, path):
        if node in path:
            if node == start and len(path) > 1:  # Valid cycle found
                cycle = path[:]  # Copy path
                min_index = cycle.index(min(cycle))  # Normalize order
                cycle = tuple(cycle[min_index:] + cycle[:min_index])
                cycles.add(cycle)
            return

        path.append(node)
        for neighbor in graph[node]:
            dfs(neighbor, start, path)
        path.pop()

    # Run DFS from each node
    for currency in graph:
        dfs(currency, currency, [])

    return [list(cycle) for cycle in cycles]

def calculate_profit(cycles, currencies, exch_rates):
    '''
    Reads in a list of cycles and outputs the total profit and a dictionary with keys of profitable
    cycles represented as a tuple and values as the corresponding profit.

    Args:
        cycles (List[List[str]]): List of cycles made in an exchange.
        currencies (List[str]): List of currencies.
        exch_matrix (nparray[nparray[float]]): A 2x2 NumPy array containing float exchange rates.

    Returns:
        float: Total profit made represented in decimal notation (Ex: 0.045 to represent 4.5%)
        Dict[tuple[str, ...]: float]: Cycles and their corresponding profits.
        
    '''
    outputs = {}

    for cycle in cycles:
        profit = 1
        names = []
        # Calculating profit in given cycle
        for i in range(len(cycle) - 1):
            names.append(cycle[i] + "/" + cycle[i + 1])
            profit *= exch_rates[currencies.index(cycle[i])][currencies.index(cycle[i + 1])]

        # Accounting for final exchange from last currency to first
        profit *= exch_rates[currencies.index(cycle[-1])][currencies.index(cycle[0])]
        names.append(cycle[-1] + "/" + cycle[0])

        # Assigning profit to cycle
        outputs[tuple(names)] = profit

    return outputs

def solve_model(model, num_chains, num_engines, Tmin, Tmax, timeout_in_secs, penalty_scaling):
    """
    Solves the currency arbitrage TitanQ model and returns the solution.

    Args:
        model (titanq.Model): The model object from the TitanQ solver containing the objective
        and constraint matrices.
        num_chains (int): Number of parallel chains running computations within each engine.
        num_engines (int): Number of independent parallel problems to run.
        Tmin (float): Minimum temperature for temperature ladder.
        Tmax (float): Maximum temperature for temperature ladder.
        timeout_in_secs (float): Maximum runtime of the solver in seconds.
        penalty_scaling (float): Scales the impact of constraint violations.

    Returns:
        TitanQ response object.

    """

    beta = (1/np.linspace(Tmin, Tmax, num_chains, dtype=np.float32)).tolist()

    # TitanQ solve
    return model.optimize(beta = beta, 
                              timeout_in_secs=timeout_in_secs, 
                              num_chains=num_chains, 
                              num_engines=num_engines, 
                              penalty_scaling=penalty_scaling)

def analyze_results(response, edge_names, currencies, exch_rate_matrix):
    """
    Analyzes and prints the results from the TitanQ solver for the currency arbitrage problem.

    Args:
        response (titanq.OptimizeResponse): The response object from TitanQ solver containing the results.
        edge_names (List[str]): List of edges in the graph.
        currencies (List[str]): List of S&P 500 stock symbols.
        exch_rate_matrix (nparray[nparray[float]]): A 2x2 NumPy array containing float exchange rates.

    Returns:
        The best cycle in the solution as a tuple.

    Prints:
    - Analysis for each solution including:
        - Engine number
        - Best cycle if it exists and is profitable
        - Product of exchange rates of best cycle
        - Profit of best cycle
        - Ising energy
    - Best overall solution
    """
    best_weight = 0
    best_cycle = ()
    # Keep track of the index and the weight of the best solution
    best_idx = -1
    print("-------- ALL ENGINE RESULTS --------")
    for idx, solution in enumerate(response.x):
        print(f"\n--- Engine {idx + 1} ---")

        # Edges used in solution
        exchanges = [edge_names[i] for i, val in enumerate(solution) if val > 0.5]

        # If no edges used
        if not exchanges:
            print("No cycle found")
            continue

        # Calculate total profit from all cycles in solution
        cycles = find_exchange_cycles(exchanges)
        
        outputs = calculate_profit(cycles, currencies, exch_rate_matrix)

        # If no exchange cycle or arbitrage opportunity exists
        if all(value < 1 for value in outputs.values()):
            print("No profitable cycle found")
            continue

        i = 1
        # Print results if a profitable cycle is found
        for cycle, profit in outputs.items():
            if profit > 1:
                print(f"Cycle {i}: {cycle}")
                print("Product of Exchange Rates: ", profit)
                print(f"Profit: {(profit - 1)* 100} %\n")
                if profit > best_weight:
                    best_idx = idx
                    best_weight = profit
                    best_cycle = cycle
                i += 1

        # Ising energy        
        print("Energy:", response.computation_metrics().get('solutions_objective_value')[idx])

    # Print the results of the best valid solution
    print("\n-------- BEST VALID SOLUTION --------")
    if best_idx == -1 or best_weight < 1:
        print("None of the engines returned profitable solutions!")
        print("Try adjusting the hyperparameters further to yield some valid solutions.")
        
    else:
        solution = response.x[best_idx]
        print(f"--- Engine {best_idx + 1} ---")
        print(f"Best Cycle: ", best_cycle)
        print("Product of Exchange Rates: ", best_weight)
        print("Best Profit: ", (best_weight - 1)*100, "%")
        print("Energy:", response.computation_metrics().get('solutions_objective_value')[best_idx])

    return best_weight, best_cycle

def plot_graph(G, best_cycle, edge_names, exch_rate_values):
    """
    Plots the currency graph with the most profitable cycle highlighted in red.

    Args:
        G (nx.MultiDiGraph): Graph object to plot currencies and exchanges on.
        best_cycle (Tuple[str]): Best cycle from the TitanQ solution.
        edge_names (List[str]): List of edges in the graph.
        exch_rate_values (np.ndarray[float]): Corresponding exchange rates to edge_names.

    Returns:
        None

    Plots:
        Graph depicting currencies as nodes and exchange rates as directed edges.
    """
    red_edges = []
    black_edges = []

    # Add edges to graph
    for index, edge in enumerate(edge_names):
        i = edge[:3]
        j = edge[4:]
        G.add_edge(i, j, weight=exch_rate_values[index])
        if edge in best_cycle:
            red_edges.append((i, j))
        else:
            black_edges.append((i, j))

    red_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True) if (u, v) in red_edges}
    black_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True) if (u, v) not in red_edges}

    pos = nx.kamada_kawai_layout(G)

    # Plot currencies as nodes
    nx.draw_networkx_nodes(G, pos, node_size = 750)

    # Plot exchange rates as directed edges
    nx.draw_networkx_edges(G, pos, node_size=750, edgelist=red_edges, edge_color='red', connectionstyle='arc3,rad=0.06', arrowstyle='-|>', arrowsize=6)
    nx.draw_networkx_edges(G, pos, node_size=750, edgelist=black_edges, edge_color='black', connectionstyle='arc3,rad=0.06', arrowstyle='-|>', arrowsize=5)

    # Labelling edges
    nx.draw_networkx_edge_labels(G, pos, edge_labels=red_labels, font_color='red', font_size=6, connectionstyle='arc3,rad=0.08')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=black_labels, font_color='black', font_size=5.1, connectionstyle='arc3,rad=0.08', alpha=0.95, label_pos=0.42)

    # Labelling nodes
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="white", font_weight="bold")
    pylab.show()

def generate_minute_profit(TITANQ_DEV_API_KEY, freq, bps):
    """
    Returns the profit of the best arbitrage opportunities on January 2, 2025 in time step increments.

    Args:
        TITANQ_DEV_API_KEY (str): TitanQ API key.
        freq (str): Frequency of time steps ('10T' is every 10 minutes, '2H' is every 2 hours).

    Returns:
        List[Timestamp]: Time steps throughout the given day at given frequency.
        List[float]: Profit of best arbitrage cycle for each corresponding time step.
    """
    # Read in data for January 2, 2025
    print("Reading data...")
    try:
        df = pd.read_csv(f"./instances/2025-01-02.csv.gz")

    except:
        print("Date is invalid.")
        return
    
    # Reformatting time rounded to the minute
    df["time"] = pd.to_datetime(df["window_start"]).dt.floor("min")
    # List of all timesteps in the day
    times = pd.date_range(start=df["time"].min(), end=df["time"].max(), freq=freq)
    profits = []

    print("Calculating (may take a few minutes)...")
    for time in times:
        # Only using currency pairs with information at that timestep
        df_time = df[df["time"] == time]
        df_time = df_time.drop_duplicates(subset="ticker", keep="last")

        # If no exchange data is available for this timestep then set to 0
        if len(df_time) == 0:
            times.append(time)
            profits.append(0)
        
        else:
            currency_pairs = df_time["ticker"].values
            
            currencies = []
            # Store all currencies
            for pair in currency_pairs:
                base, quote = pair[2:].split("-")
                currencies.append(base)
                currencies.append(quote)

            currencies = sorted(np.unique(currencies))
            size = len(currencies)
            exch_rates = pd.DataFrame(index=currencies, columns=currencies)

            # Populating exchange rate matrix
            for _, row in df_time.iterrows():
                base, quote = row["ticker"][2:].split("-")
                exch_rates.at[base, quote] = row["open"]
            
            # Exchange rates accounting for brokerage fees
            exch_rates = exch_rates.where(pd.notna(exch_rates * (1 - bps/1e4)), 0.)
            exch_rate_matrix = exch_rates.to_numpy().astype(np.float32)

            # List existing edges for TitanQ formulation
            edges = [(i, j) for i in range(size) for j in range(size) if exch_rate_matrix[i,j] > 0]
            num_edges = len(edges)

            # Exchange rates for existing edges only
            exch_rate_values = exch_rate_matrix[tuple(zip(*edges))]

            # Bias vector for objective
            bias = np.where(exch_rate_values > 0, -np.log(exch_rate_values), 1.).astype(np.float32)

            edge_names = [currencies[i]+"/"+currencies[j] for (i,j) in edges]

            model = Model(
                api_key = TITANQ_DEV_API_KEY
            )

            # Defining variables, objective, and constraints in TitanQ model
            x = model.add_variable_vector('x', num_edges, Vtype.BINARY)
            model.set_objective_expression(np.dot(bias, x), Target.MINIMIZE)

            # Cycle constraints
            for i in range(len(currencies)):
                # List of all edge indices (i, j)
                incoming = [index for index, (a, b) in enumerate(edges) if a == i]

                # List of all edge indices (j, i)
                outgoing = [index for index, (b, a) in enumerate(edges) if a == i]

                if incoming == [] and outgoing == []:
                    continue

                # Flow constraint
                model.add_constraint_from_expression(sum(x[j] for j in incoming) - sum(x[j] for j in outgoing) == 0)

                # Number of exchanges constraint
                if incoming == []:
                    model.add_constraint_from_expression(sum(x[j] for j in outgoing) <= 1)
                else:
                    model.add_constraint_from_expression(sum(x[j] for j in incoming) <= 1)

            # TitanQ Solver hyperparameters
            num_chains = 8
            num_engines = 1
            Tmin = 0.0001
            timeout_in_secs = 2
            Tmax = 550
            penalty_scaling = 45

            response = solve_model(model, 
            num_chains, num_engines, Tmin, Tmax, timeout_in_secs, penalty_scaling)

            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull):
                    # Generate best profitable cycle and corresponding profit for time step
                    best_weight, _ = analyze_results(response, edge_names, currencies, exch_rate_matrix)
                    profits.append((best_weight - 1) * 100)

    print("Done!")
    return times, profits