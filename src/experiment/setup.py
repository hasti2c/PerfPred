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

SIZE_VARS = [V.TRAIN_SIZE] if U.EXPERIMENT_TYPE == "one stage" else [V.TRAIN1_SIZE, V.TRAIN2_SIZE]
DOMAIN_VARS = [V.TRAIN_JSD] if U.EXPERIMENT_TYPE == "one stage" else [V.TRAIN1_JSD, V.TRAIN2_JSD]
LANG_VARS = [V.FEA_DIST, V.INV_DIST, V.PHO_DIST, V.SYN_DIST, V.GEN_DIST, V.GEO_DIST]

VARS_LIST, SPLITS_LIST = [], []

MODELS = {
    "linear":  {'f': F.linear},
    "poly2":   {'f': F.polynomial, 'k': 2},
    "poly3":   {'f': F.polynomial, 'k': 3},
    "exp":     {'f': F.exponential, 'bounds': (-np.inf, 6)},
    "log":     {'f': F.logarithmic},
    "power":   {'f': F.power, 'bounds': (-140, 80)},
    "mult":    {'f': F.multiplicative, 'bounds': (-100, 75)},
    "hmult":   {'f': F.hybrid_multiplicative, 'bounds': (-80, 90)},
    "am":      {'f': F.arithmetic_mean_linear, 'n': 1},
    "gm":      {'f': F.geometric_mean_linear, 'n': 1},
    "hm":      {'f': F.harmonic_mean_linear, 'n': 1},
    "scaling": {'f': F.scaling_law, 'n': 2, 'bounds': [(-np.inf, np.inf), (0, np.inf), (-85, np.inf)]},
    "anthony": {'f': F.anthonys_law, 'n': 4},
    "diff":    {'f': F.linear_with_difference, 'n': 3}
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

def init_setup():
    global VARS_LIST, SPLITS_LIST
    for k in range(U.MAX_NSPLITS + 1):
        SPLITS_LIST += map(list, list(combinations(V.main(), k)))
    
    for k in range(1, U.MAX_NVARS + 1):
        VARS_LIST += map(list, list(combinations(SIZE_VARS, k)))
        VARS_LIST += map(list, list(combinations(DOMAIN_VARS, k)))
        VARS_LIST += map(list, list(combinations(LANG_VARS, k)))

init_setup()

FULL_VARS_LIST = VARS_LIST + [SIZE_VARS + DOMAIN_VARS + LANG_VARS, SIZE_VARS, DOMAIN_VARS, LANG_VARS]

BASELINES = [
    (SIZE_VARS + DOMAIN_VARS + LANG_VARS, [], 'linear'),
    (SIZE_VARS, [], 'linear'),
    (DOMAIN_VARS, [], 'linear'),
    (LANG_VARS, [], 'linear')
]

TRIALS = pd.DataFrame(columns=["vars", "splits", "model", "trial"])

def get_path(vars, splits, model=""):
    return os.path.join(U.DATA_PATH, "results", "multi" if len(vars) > 1 else "", V.list_to_str(vars), V.list_to_str(splits), model)

def init_trial(vars, splits, model, attrs={}):
    split_names, var_names = V.list_to_str(splits), V.list_to_str(vars)
    args = MODELS[model].copy()
    if "n" not in args:
        args["n"] = len(vars)
    model_obj = M.get_instance(**args)
    
    try:
        trial = Tr(vars, splits, model_obj, get_path(vars, splits, model), model)
    except ValueError:
        return None
    row = {"vars": var_names, "splits": split_names, "model": model, "trial": trial}
    row.update(attrs)
    TRIALS.loc[len(TRIALS.index)] = row

def init_trials(vars_list=VARS_LIST, splits_list=SPLITS_LIST, models=MODELS, conditions=MODEL_CONDITIONS, 
                attrs={}):
    for vars, splits, model in list(product(vars_list, splits_list, models)):
        if model in conditions and not conditions[model](vars):
            continue
        init_trial(vars, splits, model, attrs)
    for vars, splits, model in BASELINES:
        if (vars, splits, model) not in list(product(vars_list, splits_list, models)):
            init_trial(vars, splits, model, attrs)


def get_trials(vars_list=FULL_VARS_LIST, splits_list=SPLITS_LIST, models=MODELS):
    df = TRIALS.loc[TRIALS["splits"].isin(map(V.list_to_str, splits_list))].copy()
    df = df.loc[df["vars"].isin(map(V.list_to_str, vars_list))]
    df = df.loc[df["model"].isin(models)]
    return df

init_trials()