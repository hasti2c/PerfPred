def is_pareto_dominant(mat1, mat2):
    """
    Check if mat1 pareto dominates mat2
    """
    for i in range(len(mat1)):
        if mat1[i] < mat2[i]:
            return False
    return True

def pareto(k_mats):
    """
    Take in a list of K matrices
    Returns a list of indices of Pareto optimal matrices
    """
    pareto_optimals = []

    for i in range(len(k_mats)):
        is_pareto_optimal = True

        for j in range(len(k_mats)):
            if i != j and is_pareto_dominant(k_mats[j], k_mats[i]):
                is_pareto_optimal = False
                break

        if is_pareto_optimal:
            pareto_optimals.append(i)

    return pareto_optimals

def rawlsian(k_mats):
    """
    Take in a list of K matrices
    Returns the index of the matrix with the lowest maximum entry
    """
    best_idx = 0 
    lowest_max_entry = float('inf')

    for i in range(len(k_mats)):
        cur_max_entry = max(k_mats[i].flatten()) 

        if cur_max_entry < lowest_max_entry:
            lowest_max_entry = cur_max_entry
            best_idx = i

    return best_idx
