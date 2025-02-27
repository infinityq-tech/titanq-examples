# Copyright (c) 2025, InfinityQ Technology Inc.

from collections import defaultdict

def read_instance(path):
    """
    Read in list of tickers from the file path.

    Args:
        path (str): The relative file path of the list of tickers.

    Returns:
        List[str]: List of tickers displayed row by row in the file.
    """
    with open(path) as f:
        lines = [line.rstrip().split() for line in f]
        currencies = [line[0] for line in lines]
    
    return currencies

def pair_index(name, currencies):
    '''
    Reads in name of exchange rate in format "XXXYYY=X" where "XXX" is the first currency and "YYY" is the second,
    and a list of currencies, and outputs the index of the exchange rate in the flattened exchange rate vector.

    Args: 
        name (str): Name of currency exchange
        currencies (nparray[str]): List of currencies in the order they appear in the exchange matrix rows & columns.

    Returns:
        int: Index of exchange rate in flattened 1D vector.
    '''
    return currencies.index(name[:3])*len(currencies)+currencies.index(name[3:6])

def find_exchange_cycles(exchanges):
    '''
    Reads in list of exchanges made in format "XXX/YYY" where "XXX" is the first currency and "YYY" is the second,
    and outputs a dictionary where each key is a cycle that maps to the profit made in that cycle. 
    Prints the different cycles.

    Args:
        exchanges (nparray[str]): List of exchanges made.
        currencies (nparray[str]): List of currencies in the order they appear in the exchange matrix rows & columns.

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
    Reads in a list of cycles and outputs the total profit made.

    Args:
        cycles (list[list[str]]): List of cycles made in an exchange.
        currencies (list[str]): List of currencies.
        exch_matrix (nparray[nparray[float]]): A 2x2 NumPy array containing float exchange rates.

    Returns:
        float: Total profit made represented in decimal notation (Ex: 0.045 to represent 4.5%)
        Dict: Cycles and their corresponding profits.
            tuple[str, ...]: float
        
    '''

    solution_profit = 0
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

        # Adding cycle profit to total profit
        solution_profit += profit

    return (solution_profit - len(cycles)) / len(cycles) if cycles else 0, outputs