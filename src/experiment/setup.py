import os
import typing as Typ
from itertools import combinations, product

import numpy as np
import pandas as pd

import modeling.functions as F
import util as U
from modeling.model import Model as M
from modeling.trial import Trial as T
from slicing.variable import Variable as V

SIZE_VARS = [V.TRAIN_SIZE] if U.EXPERIMENT_TYPE == "one stage" else [V.TRAIN1_SIZE, V.TRAIN2_SIZE]
DOMAIN_VARS = [V.TRAIN_JSD] if U.EXPERIMENT_TYPE == "one stage" else [V.TRAIN1_JSD, V.TRAIN2_JSD]
LANG_VARS = [V.FEA_DIST, V.INV_DIST, V.PHO_DIST, V.SYN_DIST, V.GEN_DIST, V.GEO_DIST]
ALL_VARS = SIZE_VARS + DOMAIN_VARS + LANG_VARS

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

def init_setup() -> None:
    """ Initialize VARS_LIST and SPLITS_LIST based on MAX_NVARS and MAX_NSPLITS. """
    global VARS_LIST, SPLITS_LIST
    for k in range(U.MAX_NSPLITS + 1):
        SPLITS_LIST += map(list, list(combinations(V.main(), k)))
    
    for k in range(1, U.MAX_NVARS + 1):
        VARS_LIST += map(list, list(combinations(SIZE_VARS, k)))
        VARS_LIST += map(list, list(combinations(DOMAIN_VARS, k)))
        VARS_LIST += map(list, list(combinations(LANG_VARS, k)))

init_setup()

BASELINE_VARS_LIST = [SIZE_VARS + DOMAIN_VARS, LANG_VARS, SIZE_VARS + DOMAIN_VARS + LANG_VARS]
FULL_VARS_LIST = VARS_LIST + BASELINE_VARS_LIST

TRIALS = pd.DataFrame(columns=["vars", "splits", "model", "trial"])

def get_path(vars: Typ.Union[str, list[V]]="", splits: Typ.Union[str, list[V]]="", model: str="") -> str:
    """ Returns path of experiment/trial given vars, splits, and optionally model."""
    return os.path.join(U.DATA_PATH, "results", "multi" if len(vars) > 1 else "", 
                        vars if isinstance(vars, str) else V.list_to_str(vars), 
                        splits if isinstance(splits, str) else V.list_to_str(splits), model)

def init_trial(vars: list[V], splits: list[V], model: str) -> None:
    """ Given vars, splits and models, creates corresponding trial and adds it to TRIALS. """
    split_names, var_names = V.list_to_str(splits), V.list_to_str(vars)
    args = MODELS[model].copy()
    if "n" not in args:
        args["n"] = len(vars)
    model_obj = M.get_instance(**args)
    
    try:
        trial = T(vars, splits, model_obj, get_path(vars, splits, model), model)
    except ValueError:
        return None
    TRIALS.loc[len(TRIALS.index)] = {"vars": var_names, "splits": split_names, "model": model, "trial": trial}

def init_trials() -> None:
    """ Initializes TRIALS based on vars_list, splits_list, models, and conditions. """
    for vars, splits, model in list(product(VARS_LIST, SPLITS_LIST, MODELS)):
        if model not in MODEL_CONDITIONS or MODEL_CONDITIONS[model](vars):
            init_trial(vars, splits, model)
    for vars, splits, model in list(product(BASELINE_VARS_LIST, SPLITS_LIST, MODELS)):
        if model not in MODEL_CONDITIONS or MODEL_CONDITIONS[model](vars):
            init_trial(vars, splits, model)

def get_trials(vars: Typ.Optional[str]=None, splits: Typ.Optional[str]=None, model: Typ.Optional[str]=None, 
               df: pd.DataFrame=TRIALS) -> pd.DataFrame:
    """ Returns subset of TRIALS with vars, splits, and model within the specified values. """
    if vars is not None:
        df = df.loc[df["vars"] == vars]
    if splits is not None:
        df = df.loc[df["splits"] == splits]
    if model is not None:
        df = df.loc[df["model"] == model]
    return df.reset_index()

def find_trial(vars: str, splits: str, model: str) -> T:
    df = get_trials(vars, splits, model)
    if len(df) > 1:
        raise ValueError("Arguments don't specify a unique trial.")
    elif len(df) == 0:
        raise ValueError("No trial matches the arguments.")
    return df.iloc[0].loc["trial"]

init_trials()