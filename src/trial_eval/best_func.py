import trial_eval.tf_eval as E

def is_pareto_dominant(vec1, vec2):
    """
    Check if mat1 pareto dominates mat2
    """
    for i in range(len(vec1)):
        if vec1[i] < vec2[i]:
            return False
    return True

def pareto(cost_vecs):
    """
    Take in a list of K matrices
    Returns a list of indices of Pareto optimal matrices
    """
    pareto_optimals = []

    for i in range(len(cost_vecs)):
        is_pareto_optimal = True

        for j in range(len(cost_vecs)):
            if i != j and is_pareto_dominant(cost_vecs[j], cost_vecs[i]):
                is_pareto_optimal = False
                break

        if is_pareto_optimal:
            pareto_optimals.append(i)

    return pareto_optimals

def rawlsian(cost_vecs):
    """
    Take in a list of K matrices
    Returns the index of the matrix with the lowest maximum entry
    """
    best_idx = 0 
    lowest_max_entry = float('inf')

    for i in range(len(cost_vecs)):
        cur_max_entry = max(cost_vecs[i]) 

        if cur_max_entry < lowest_max_entry:
            lowest_max_entry = cur_max_entry
            best_idx = i

    return best_idx

# ----------------SHOULD NOT BE HERE -------------------

trials = [] # Somehow have a list of trials

def whos_da_best(trials):
    
    best_trials = []
    
    for approach in E.MRF:
        cost_vecs = []
        for trial in trials:
            cost_vecs.append(trial.cost_vec(MRF = approach))
        p_indices = pareto(cost_vecs)
        best_trials.append(rawlsian(cost_vecs[p] for p in p_indices))
    
    if (all(best == best_trials[0] for best in best_trials)):
        print (best_trials[0].name + " is the winner!")
    
    print("Best for simple average = " + best_trials[0].name + 
          "\n Best for best set of fits = " + best_trials[1].name + 
          "\n Best for cross average fits = " + best_trials[2].name)
    
    