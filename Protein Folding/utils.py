import copy
import sympy as sp
import numpy as np

def reduce_min(f, num_vars, variable_label = 'x', starting_index = 0):
    """Reduces a Higher-Order Binary function to a Quadratic function.
    
    Reference paper : https://www.sciencedirect.com/science/article/pii/S0166218X01003419?ref=pdf_download&fr=RR-2&rr=88f867a12e904bd0

    Examples: 

    Let's assume we want to reduce the following polynomial expression: :math: `f(x1,x2,x3) = x1*x2*x3`.

        >>> f = {('x1','x2','x3'): 1}
        >>> g = reduce_min(f, num_vars = 3, variable_label = 'x', starting_index = 1)
        >>> print(g)
        {('x1','x2'): 3, ('x1','x4'): -6, ('x2','x4'): -6,('x4',): 9, ('x3','x4'): 1}

    Args:
        f (Dict{Tuple[str]: float}): The dictionary representing the Higher-Order function to reduce.
            The key is the string tuple representing the term of the function.
            The value is the bias corresponding to the term.
        num_vars (int): The number of variables in the input function.
        variable_label (str, optional): The label of the variable used in the input function. Defaults to 'x'.
        starting_index (int, optional): The starting index of the variables of the input function. Defaults to 0.

    Returns:
        Dict{Tuple[str]: float}: The dictionary representing the Quadratic function after the reduction of the input function.
            The key is the string tuple representing the term of the function.
            The value is the bias corresponding to the term.
    """

    M = 1 + 2 * sum(map(abs, f.values()))
    m = num_vars
    g = copy.deepcopy(f)

    S_star = check_subsets_with_higher_order(g)
    while S_star:
        h = copy.deepcopy(g)
        i,j = sorted(tuple((S_star[0],S_star[1])))

        if (i,j) in g.keys():
            h[(i,j)] = g[(i,j)] + M
        elif (j,i) in g.keys():
            h[(i,j)] = g[(j,i)] + M
        else:
            h[(i,j)] = M

        h[(i,f'{variable_label}{m + starting_index}')] = -2 * M
        h[(j,f'{variable_label}{m + starting_index}')] = -2 * M
        h[(f'{variable_label}{m + starting_index}',)] = 3 * M

        for key,val in g.items():
            if i in key and j in key and val != 0 and (i,j) != key and (j,i) != key:
                vars = list(key)
                vars.remove(i)
                vars.remove(j)
                vars.append(f'{variable_label}{m + starting_index}')
                subset = tuple(sorted(tuple(vars),key= lambda s : int(s[1:])))
                h[subset] = h[key]
                h[key] = 0

        m += 1
        g = copy.deepcopy(h)
        del g[S_star]
        g = {k: v for k, v in g.items() if v != 0}
        g = update_dict_with_unique_tuples(g)
        S_star = check_subsets_with_higher_order(g)

    non_zeros = {k: v for k, v in g.items() if v != 0}

    # Decompose to linear and quadratic
    linear = {}
    quadratic = {}
    for key,val in non_zeros.items():
        if len(key) == 1:
            linear[key[0]] = val
        else:
            quadratic[key] = val
    return linear,quadratic,m,non_zeros

def check_subsets_with_higher_order(f):
    """Checks if the input function contains a higher-order (cubic, quartic, etc. ) term and returns its key if it exists.

     Examples:

        >>> f = {('x1','x2'): 5, ('x1','x3'): 2}
        >>> g = check_subsets_with_higher_order(f)
        >>> print(g)
        False
        >>> f = {('x1','x2'): 5, ('x1','x3','x4'): 2}
        >>> g = check_subsets_with_higher_order(f)
        >>> print(g)
        ('x1','x3','x4')

    Args:
        f (Dict{Tuple[str]: float}): The dictionary representing the Higher-Order function to reduce.
            The key is the string tuple representing the term of the function.
            The value is the bias corresponding to the term.

    Returns:
        Union[Boolean, Tuple[str]]: 
            - A boolean if the higher-order term doesn't exist. 
            - The higher-order term as a tuple. 
    """

    for key,_ in f.items():
        if len(key) > 2:
            return key
    return False

def update_dict_with_unique_tuples(f):
    """Fixes the expression that has duplicate terms by combining them as a summation.

    Examples:

        >>> f = {('x1','x2'): 5, ('x1','x2'): 2}
        >>> g = update_dict_with_unique_tuples(f)
        >>> print(g)
        {('x1','x2'): 7}

    Args:
        f (Dict{Tuple[str]: float}): The polynomial dictionary to fix.
            The key is the strings tuple representing the term of the function.
            The value is the bias corresponding to the term.

    Returns:
        f (Dict{Tuple[str]: float}): The dictionary of the input expression with no duplicates.
    """

    updated_dict = {}
    for key, value in f.items():
        unique_tuple = tuple(sorted(set(key)))
        
        if unique_tuple in updated_dict:
            updated_dict[unique_tuple] += value
        else:
            updated_dict[unique_tuple] = value
    
    return updated_dict

def transform_polynomial(polynomial, vars, label, f = lambda x: 2* x -1):
    """Takes any polynomial and makes the change of variable described in the parameter ``f``.

    Args:
        polynomial (Dict{Tuple[str]: float}): The dictionary representing the polynomial function on which to do the change of variable.
            The key is the string tuple representing the term of the function.
            The value is the bias corresponding to the term.
        vars (List[str]): List of variable labels of the input polynomial.
        label (str): The label of the new variable for the output polynomial after the change of variable.
        f (lambda, optional): The lambda function describing the change of variable to do on the input polynomial. Defaults to lambda x:2*x-1.

    Returns:
        Dict{Tuple[str]: float}: The polynomial corresponding to the input polynomial after making the change of variable.
    """
    
    # Transform variables to sympy.Symbols
    sympy_vars = sp.symbols(' '.join(vars))

    # Define the transformation
    new_vars = [sp.Symbol(f'{label}{i}') for i in range(len(sympy_vars))]
    transformations = {z: f(x) for z, x in zip(sympy_vars, new_vars)}
    
    # Convert the input dictionary to a sympy expression
    polynomial = sum(coeff * sp.Mul(*sp.symbols(monomial)) for monomial, coeff in polynomial.items())
    
    # Apply the transformation
    transformed_polynomial = polynomial.subs(transformations)
    
    # Expand the transformed polynomial
    expanded_polynomial = sp.expand(transformed_polynomial)
    
    # Convert the expanded polynomial to a dictionary
    poly_dict = expanded_polynomial.as_coefficients_dict()

    # Format the dictionary keys as tuples
    formatted_dict = {}
    for term, coeff in poly_dict.items():
        if term == 1:
            formatted_dict[()] = coeff
        else:
            variables = sp.Mul.make_args(term)
            formatted_dict[tuple(sorted(map(lambda x:x.name, variables), key=str))] = coeff
    
    return formatted_dict

def from_pauli_to_bipolar(string):
    """Takes the Pauli term as a string of Pauli matrices Z and I, and transform it to a bipolar term using bipolar variables z_i.

    Reference paper: https://arxiv.org/pdf/1706.02998

    Examples:

    Let's consider the term "IZIZZI" composed of Pauli Matrices on the Z-axis. The following will transform it to a bipolar term.

        >>> pauli_term = "IZIZZI"
        >>> bipolar_term = from_pauli_to_bipolar(pauli_term)
        >>> print(g)
        ('z1','z3','z4')

    Args:
        string (str): The string representing the Pauli term where each character is a Pauli matrix I or Z.

    Returns:
        Tuple[str]: A tuple of bipolar variable labels encoding the Pauli term.
    """

    labels_array = []
    # Returning the indices of Pauli Z
    ids = np.where(np.array([*string])=="Z")[0]
    for id in ids:
        labels_array.append(f"z{id}")
    return tuple(labels_array)

def extract_pauli_strings_and_coeffs(pauli_sum_op):
    """Extracts the Pauli strings and their corresponding coefficients from a PauliSumOp object.

    Example:

        >>> from qiskit.opflow import PauliSumOp
        >>> from qiskit.quantum_info import SparsePauliOp
        >>> sparse_pauli_op = SparsePauliOp(['IIIIIIIII', 'IIIIIIZII'], coeffs=[1.0, -0.5])
        >>> pauli_sum_op = PauliSumOp(sparse_pauli_op)
        >>> pauli_strings, coeffs = extract_pauli_strings_and_coeffs(pauli_sum_op)
        >>> for ps, coeff in zip(pauli_strings, coeffs):
        >>>     print(f"Pauli string: {ps}, Coefficient: {coeff}")
        Pauli string: IIIIIIIII, Coefficient: 1.0
        Pauli string: IIIIIIZII, Coefficient: -0.5

    Args:
        pauli_sum_op (PauliSumOp): The PauliSumOp object containing Pauli operators and their coefficients.

    Returns:
        pauli_strings (List[str]): A list of Pauli strings representing the Pauli operators.
        coeffs (List[float]): An array of coefficients corresponding to the Pauli operators.
    """
    # Get the SparsePauliOp
    sparse_pauli_op = pauli_sum_op.primitive

    # Get the Pauli strings
    pauli_strings = sparse_pauli_op.paulis.to_labels()

    # Get the coefficients
    coeffs = sparse_pauli_op.coeffs.real

    return zip(pauli_strings, coeffs)