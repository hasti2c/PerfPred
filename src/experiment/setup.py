import os
import sys
from itertools import product, combinations

import numpy as np
import pandas as pd

import modeling.functions as F
import util as U
from modeling.model import Model as M
from modeling.trial import Trial as Tr
from slicing.variable import Variable as V

VARS, SPLITS = [], []

MODELS = {
    "linear":      {'f': F.linear},
    "poly2":       {'f': F.polynomial, 'k': 2},
    "poly3":       {'f': F.polynomial, 'k': 3},
    "exp":         {'f': F.exponential, 'bounds': (-np.inf, 6)},
    "log":         {'f': F.logarithmic},
    "power":       {'f': F.power, 'bounds': (-140, 80)},
    "mult":        {'f': F.multiplicative, 'bounds': (-100, 75)},
    "hybrid_mult": {'f': F.hybrid_multiplicative, 'bounds': (-80, 90)},
    "am":          {'f': F.arithmetic_mean_linear, 'n': 1},
    "gm":          {'f': F.geometric_mean_linear, 'n': 1},
    "hm":          {'f': F.harmonic_mean_linear, 'n': 1},
    "scaling":     {'f': F.scaling_law, 'n': 2, 'bounds': [(-np.inf, np.inf), (0, np.inf), (-85, np.inf)]},
    "anthony":     {'f': F.anthonys_law, 'n': 4},
    "diff":        {'f': F.linear_with_difference, 'n': 3}
}

MODEL_CONDITIONS = {
    "mult":    lambda vars: len(vars) > 1,
    "am":      lambda vars: len(vars) > 1,
    "gm":      lambda vars: len(vars) > 1,
    "hm":      lambda vars: len(vars) > 1,
    "scaling": lambda vars: vars == [V.TRAIN_SIZE] if U.EXPERIMENT_TYPE == "one stage" 
                                                 else vars in [[V.TRAIN1_SIZE], [V.TRAIN2_SIZE]],
    "anthony": lambda vars: U.EXPERIMENT_TYPE == "two stage" and vars == [V.TRAIN1_SIZE, V.TRAIN2_SIZE],
    "diff":    lambda vars: len(vars) == 2
}

TRIALS = pd.DataFrame(columns=["vars", "splits", "model", "trial"])

def init_setup():
    global VARS, SPLITS
    for k in range(U.MAX_NSPLITS + 1):
        SPLITS += map(list, list(combinations(V.main(), k)))

    size_vars = [V.TRAIN_SIZE] if U.EXPERIMENT_TYPE == "one stage" else [V.TRAIN1_SIZE, V.TRAIN2_SIZE]
    domain_vars = [V.TRAIN_JSD] if U.EXPERIMENT_TYPE == "one stage" else [V.TRAIN1_JSD, V.TRAIN2_JSD]
    lang_vars = [V.FEA_DIST, V.INV_DIST, V.PHO_DIST, V.SYN_DIST, V.GEN_DIST, V.GEO_DIST]
    for k in range(1, U.MAX_NVARS + 1):
        VARS += map(list, list(combinations(size_vars, k)))
        VARS += map(list, list(combinations(domain_vars, k)))
        VARS += map(list, list(combinations(lang_vars, k)))

def init_trial(splits, vars, model, verbose=False):
    split_names, var_names = V.list_to_str(splits), V.list_to_str(vars)
    path = os.path.join(U.DATA_PATH, "results", "multi" if len(vars) > 1 else "", var_names, split_names, model)
    
    args = MODELS[model].copy()
    if "n" not in args:
        args["n"] = len(vars)
    model_obj = M.get_instance(**args)
    
    try:
        trial = Tr(splits, vars, model_obj, path, model)
    except ValueError:
        return None
    if verbose:
        print(f"Initialized {trial}", file=sys.stderr)
    return {"vars": var_names, "splits": split_names, "model": model, "trial": trial}

def init_all(verbose=False):
    init_setup()
    for vars, splits, model in product(VARS, SPLITS, MODELS):
        if model in MODEL_CONDITIONS and not MODEL_CONDITIONS[model](vars):
            continue
        row = init_trial(splits, vars, model, verbose)
        if row is not None:
            TRIALS.loc[len(TRIALS.index)] = row

def get_trials(splits=SPLITS, vars=VARS, models=MODELS):
    df = TRIALS
    if splits:
        df = df.loc[df["splits"].isin(map(V.list_to_str, splits))]
    if vars:
        df = df.loc[df["vars"].isin(map(V.list_to_str, vars))]
    if models:
        df = df.loc[df["model"].isin(models)]
    return df

init_all()