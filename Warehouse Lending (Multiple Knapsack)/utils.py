import math
import numpy as np

def gen_data(model_dict):
    """Generates the weight matrix W and bias vector b of the objective function:
            -1/2 * x^T.W.x - b.x
    Generates the constraint weights A and constraint bounds (c^l,c^u) of the constraint:
            c^l <= A.x <= c^u 

    Args:
        model_dict (Dict{str:Generic[T]}): The mathematical model holding the variables, the constraints, and the objective function.

    Returns:
        int: The number of decision variables of the problem.
        Numpy.array[Numpy.float32,Numpy.float32]: The weight matrix of the objective function.
        Numpy.array[Numpy.float32]: The bias vector of the objective function.
        Numpy.array[Numpy.float32,Numpy.float32]: The matrix A with constraint coefficients.
        Numpy.array[Numpy.float32,Numpy.float32]: The vector of lower and upper bounds for each constraint.
    """

    # Process constraints
    model_dict["hard_constraints"] = []

    for con in model_dict["constraints"]:
        if con[2] == "<":
            slack_var, obj_coeff = proc_inequality(con, model_dict["inequality_strength"])
            model_dict["variables"] += slack_var
            model_dict["objective"] += obj_coeff
            
        elif con[2] == "=":
            con_coeff, rhs = proc_equality(con)
            model_dict["hard_constraints"].append((con_coeff, rhs))

    model_dict["name_id"], model_dict["id_name"] = create_var_ids(model_dict["variables"])
    model_dict["con_name_id"], model_dict["id_con_name"] = create_con_ids(model_dict["hard_constraints"])

    n = len(model_dict["variables"])
    m = len(model_dict["hard_constraints"])

    b = np.zeros(n, dtype=np.float32)
    W = np.zeros((n,n), dtype=np.float32)

    W_con = np.zeros((m,n), dtype=np.float32)
    con_bounds = np.zeros((m,2), dtype=np.float32)

    for coeff in model_dict["objective"]:
        i = model_dict["name_id"][coeff[0][0]]
        j = model_dict["name_id"][coeff[0][1]]
        val = coeff[1]
        if i == j:
            b[i] += val
        else:
            W[i][j] += val
            W[j][i] += val

    for con in model_dict["hard_constraints"]:
        con_name = con[0][0][0][1]
        i = model_dict["con_name_id"][con_name]
        con_bounds[i][0] = con[1]
        con_bounds[i][1] = con[1]

        for var in con[0]:
            var_name = var[0][0]
            j = model_dict["name_id"][var_name]
            val = var[1]
            W_con[i][j] = val

    return n, W, b, W_con, con_bounds

def proc_inequality(con, strength):
    """Decodes the inequality constraint to add it to the constraint weights matrix and constraint bounds.

    Args:
        con (Tuple[str,List[Tuple[str,Numpy.float32]],str,Float32]): The tuple holding the name of the decision variable, the left-hand side, the right-hand side and the constraint operator.
        strength (float32): The strength of the constraint.

    Returns:
        List[str]: List of the new slack variables to be added to the list of decision variables of the problem.
        List[Tuple[Tuple[str,str],Numpy.float32]]: The list of tuples holding the coefficients for the weight matrix and bias vector.
    """

    rhs = con[3]
    bits = math.ceil(math.log2(rhs))
    slack_var = []

    for i in range(bits):
        name = f'{con[0]}_sv_[{i}]'
        slack_var.append(name)
        coeff = 2**i
        con[1].append((name, coeff))

    rhs = 2**bits - 1
    obj_coeff = []
    num_var = len(con[1])

    for i in range(num_var):
        name_i = con[1][i][0]
        coeff_i = con[1][i][1]

        for j in range(i+1, num_var):
            name_j = con[1][j][0]
            coeff_j = con[1][j][1]
            # strength*(sum_i(w_i * x_i) - rhs + slack)^2 --> w_i * w_j * x_i * x_j
            weight = 2 * coeff_i * coeff_j * strength
            obj_coeff.append(((name_i, name_j), weight))

        # Bias
        bias = coeff_i*(coeff_i - 2*rhs) * strength
        obj_coeff.append(((name_i, name_i), bias))


    return slack_var, obj_coeff


def proc_equality(con):
    """Decodes the equality constraint to add it to the constraint weights matrix and constraint bounds.

    Args:
        con (Tuple[str,List[Tuple[str,Numpy.float32]],str,Float32]): The tuple holding the name of the decision variable, the left-hand side, the right-hand side and the constraint operator.

    Returns:
        List[Tuple[Tuple[str,str],Numpy.float32]]: The list of tuples holding the coefficients for the weight matrix and bias vector.
        float32: The right hand side of the equality constraint.
    """

    rhs = con[3]
    con_coeff = []
    num_var = len(con[1])

    for i in range(num_var):
        name_i = con[1][i][0]
        coeff_i = con[1][i][1]
        con_coeff.append(((name_i, con[0]), coeff_i))

    return con_coeff, rhs

def create_var_ids(variables):
    """Maps each decision variable to an integer id.

    Args:
        variables (List[str]): The list of the decision variables of the problem.

    Returns:
        Dict{str:int}: The dictionary mapping the decision variable label to an integer id.
        Dict{int:str}: The dictionary mapping the decision variable id to its label.
    """

    name_id_dict = dict()
    id_name_dict = dict()
    id = 0

    for var in variables:
        name_id_dict[var] = id
        id_name_dict[id] = var
        id += 1

    return name_id_dict, id_name_dict

def create_con_ids(cons):
    """Maps each constraint to an integer id.

    Args:
        con (List[Tuple[List[Tuple[Tuple[str,str],Numpy.float32]],Numpy.float32]]): The list of the decision variables of the problem.

    Returns:
        Dict{str:int}: The dictionary mapping the constraint label to an integer id.
        Dict{int:str}: The dictionary mapping the constraint id to its label.
    """

    name_id_dict = dict()
    id_name_dict = dict()
    id = 0

    for con in cons:
        con_name = con[0][0][0][1]
        name_id_dict[con_name] = id
        id_name_dict[id] = con_name
        id += 1

    return name_id_dict, id_name_dict
