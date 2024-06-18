# Copyright (c) 2024, InfinityQ Technology Inc.

import math

class WarehouseLendingModel:
    """A class to hold the different input values and for creating the mathematical model.
    
    Args:
        new_loans (List[Dict{Generic[T]:Generic[T]}]): The list of the new loans that we want to assign to the available warehouses.
        warehouses (List[Dict{Generic[T]:Generic[T]}]): The list of warehouses that provide loans.

    """

    def __init__(self,new_loans,warehouses) -> None:
        
        self.new_loans = new_loans
        self.warehouses = warehouses
        self.num_new_loans = len(new_loans)
        self.num_warehouses = len(warehouses)
        self.valid_loan_wh = dict()
        self.valid_wh_loan = dict()
        self.model_dict = dict()


    def generate_valid_loan_warehouse_assignments(self):
        """
        Generates a valid assignment of loans to warehouses and populating the dictionary self.valid_loan_wh.
        
        Returns:
            None
        """
        
        # Check which warehouses a loan can be assigned to
        for i in range(self.num_new_loans):
            # List of valid warehouses for loan i
            valid_warehouses = [] 

            # Check each warehouse to see if they have the correct pools
            for j in range(self.num_warehouses):
                valid = True

                # Generate the list of pools in the warehouse
                wh_pools = [pool["name"] for pool in self.warehouses[j]["pools"]]

                # Check if all the loan pools exist in the warehouse
                for p in self.new_loans[i]["pools"]:
                    if p not in wh_pools:
                        valid = False
                        break
                
                if valid:
                    valid_warehouses.append(j)

            self.valid_loan_wh[i] = valid_warehouses

    def generate_valid_warehouse_loan_assignments(self):
        """
        Generates a valid assignment of the warehouses to the loans and populating the dictionary self.valid_wh_loan.
        
        Returns:
            None
        """

        for j in range(self.num_warehouses):
            self.valid_wh_loan[j] = []

            for i in range(self.num_new_loans):
                if j in self.valid_loan_wh[i]:
                    self.valid_wh_loan[j].append(i)
        
    def add_variables(self):
        """
        Adds the variables to the model self.model_dict.
        
        Returns:
            None
        """
        
        variables = []
        for i in range(self.num_new_loans):
            for wh in self.valid_loan_wh[i]:
                var_name = f'l_{i}_wh_{wh}'
                variables.append(var_name)

        self.model_dict["variables"] = variables

    def add_constraints(self):
        """
        Adds the constraints to the model self.model_dict.
        
        Returns:
            None
        """
        
        constraints = []
        # Set partitioning constraints: 
        #   - Each loan must be assigned to exactly one warehouse
        for i in range(self.num_new_loans):
            # lhs = rhs
            # sum_j(1.0*l_i_wh_j) = 1.0
                
            name = f'l_{i}'
            lhs = []
            op = "="
            rhs = 1.0
            
            # For every warehouse that the loan can be assigned to:
            for wh in self.valid_loan_wh[i]:
                var_name = f'l_{i}_wh_{wh}'
                coeff = 1.0 

                # Build a tuple with the corresponding variable name and coefficient
                # Append tuple to lhs 
                lhs.append((var_name, coeff))

            # Build constraint definition as a tuple with the format below
            con = (name, lhs, op, rhs)        

            constraints.append(con)

        # Inequality constraints: 
        #   - The percentage value of each pool relative to total loans in each warehouse < limit_pct
        for j in range(self.num_warehouses):
            for pool in self.warehouses[j]["pools"]:
                # lhs < rhs
                #
                # (sum_i(value_loan_i * l_i_wh_j) + value_wh_pool)/(sum_i( value_loan_i * l_i_wh_j) + total_loans_wh_j) < limit_pct_wh_j_pool
                #   multiply both sides by denominator of lhs
                #
                # (1-limit_pct_wh_j_pool) * sum_i( value_loan_i * l_i_wh_j) < limit_pct_wh_j_pool * total_loans_wh_j - value_wh_pool
                
                name = f'wh_{j}_{pool["name"]}'
                lhs = []
                op = "<"

                # Multiply by 100 to convert to percentage and to avoid fractional components
                rhs = 100.0*(pool["limit_pct"] * self.warehouses[j]["total_loans"] - pool["value"])

                pct_scaling = 100.0*(1.0 - pool["limit_pct"])

                # For every loan that the warehouse could receive
                for loan_i in self.valid_wh_loan[j]:

                    # if the loan belongs to the pool currently being evaluated 
                    if pool["name"] in self.new_loans[loan_i]["pools"]:
                        var_name = f'l_{loan_i}_wh_{j}' # naming consistent with initial variable creation
                        coeff = self.new_loans[loan_i]["value"]
                        coeff *= pct_scaling

                        lhs.append((var_name, coeff))
            
                # Add non-empty constraints
                if len(lhs) != 0:
                    con = (name, lhs, op, rhs)        
                    constraints.append(con)


        self.model_dict["constraints"] = constraints

    def add_objective(self):    
        """
        Adds the objective function to the model self.model_dict.
        
        Returns:
            None
        """

        objective = []
        # Objective = sum_i( scaling_factor_wh_pool * sum_j( value_loan_i * l_i_wh_j ) )
        # Scaling factor takes into account the warehouse pool's limit
        for i in range(self.num_new_loans):    
            for wh in self.valid_loan_wh[i]:
                for pool in self.warehouses[wh]["pools"]:
                    var_name = f'l_{i}_wh_{wh}'

                    pct_scaling = 100.0*(1.0 - pool["limit_pct"])

                    coeff = self.new_loans[i]["value"]
                    coeff *= pct_scaling

                    objective.append(((var_name, var_name), coeff))

        self.model_dict["objective"] = objective

    def set_constraint_strength(self):
        """
        Adds the constraints strength to the model self.model_dict.
        
        Returns:
            None
        """

        sum_loan_values = sum(loan["value"] for loan in self.new_loans)
        self.model_dict["inequality_strength"] = self.num_new_loans * math.floor(math.sqrt(sum_loan_values))