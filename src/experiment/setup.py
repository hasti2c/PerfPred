import os
import typing as Typ
from itertools import combinations, product

import numpy as np
import pandas as pd

import modeling.functions as F
import util as U
from modeling.model import Model as M
from modeling.trial import Trial as T
from slicing.slice import SliceGroup as SG
from slicing.variable import Variable as V

# Models specified by the input values of Model.get_instance()
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
    "scaling": {'f': F.scaling_law, 'n': 2, 'bounds': [(-np.inf, 120), (0, np.inf), (-85, np.inf)]},
    "anthony": {'f': F.anthonys_law, 'n': 4},
    "diff":    {'f': F.linear_with_difference, 'n': 3}
}

# Conditions specifying which trials to run model on. Models not in this dictionary are ran on all trials.
MODEL_CONDITIONS = {
    "mult":    lambda vars: len(vars) > 1,
    "am":      lambda vars: len(vars) > 1,
    "gm":      lambda vars: len(vars) > 1,
    "hm":      lambda vars: len(vars) > 1,
    "scaling": lambda vars: vars in [[V.TRAIN_SIZE], [V.TRAIN_NORM_SIZE]] if U.EXPERIMENT_TYPE == "one stage" 
                            else vars in [[V.TRAIN1_SIZE], [V.TRAIN1_NORM_SIZE], [V.TRAIN2_SIZE], [V.TRAIN2_NORM_SIZE]],
    "anthony": lambda vars: U.EXPERIMENT_TYPE == "two stage" and vars in [[V.TRAIN1_SIZE, V.TRAIN2_SIZE], 
                                                                          [V.TRAIN1_NORM_SIZE, V.TRAIN2_NORM_SIZE]],
    "diff":    lambda vars: len(vars) == 2
}

# List of vars of each type (for convenience). 
SIZE_VARS = [V.TRAIN_SIZE, V.TRAIN_NORM_SIZE] if U.EXPERIMENT_TYPE == "one stage" \
            else [V.TRAIN1_SIZE, V.TRAIN1_NORM_SIZE, V.TRAIN2_SIZE, V.TRAIN2_NORM_SIZE]
DOMAIN_VARS = [V.TRAIN_JSD] if U.EXPERIMENT_TYPE == "one stage" else [V.TRAIN1_JSD, V.TRAIN2_JSD]
LANG_VARS = [V.FEA_DIST, V.INV_DIST, V.PHO_DIST, V.SYN_DIST, V.GEN_DIST, V.GEO_DIST]
ALL_VARS = SIZE_VARS + DOMAIN_VARS + LANG_VARS # List of all values of Variable enum.

# VARS_LIST is the list of vars configurations.
# single factor
VARS_LIST = [[var] for var in ALL_VARS]
# multi factor: size+jsd, nsize+jsd, langvars, size+jsd+langvars, nsize+jsd+langvars
VARS_LIST += [[V.TRAIN_SIZE, V.TRAIN_JSD], [V.TRAIN_NORM_SIZE, V.TRAIN_JSD], LANG_VARS,
             [V.TRAIN_SIZE, V.TRAIN_JSD] + LANG_VARS, [V.TRAIN_NORM_SIZE, V.TRAIN_JSD] + LANG_VARS]

# SPLITS_LIST is the list of vars configurations.
SPLITS_LIST = []
# all combinations in which all slices have at least MIN_POINTS points
for k in range(len(V.main())):
    combs = map(list, list(combinations(V.main(), k))) # combinations of length k
    for splits in combs:
        slices = SG.get_instance(V.complement(splits))
        if min([len(slice) for slice in slices.slices]) >= U.MIN_POINTS: # check MIN_POINTS
            SPLITS_LIST.append(splits)

TRIALS = pd.DataFrame(columns=["vars", "splits", "model", "trial"])

def get_path(vars: Typ.Union[str, list[V]]="", splits: Typ.Union[str, list[V]]="", model: str="") -> str:
    """ Returns path of experiment/trial given vars, splits, and optionally model."""
    return os.path.join(U.DATA_PATH, U.RESULTS_DIR, vars if isinstance(vars, str) else V.list_to_str(vars), 
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

def get_trials(vars: Typ.Optional[Typ.Union[str, list[V]]]=None, splits: Typ.Optional[Typ.Union[str, list[V]]]=None, 
               model: Typ.Optional[str]=None, df: pd.DataFrame=TRIALS) -> pd.DataFrame:
    """ Returns subset of TRIALS with vars, splits, and model within the specified values. """
    if vars is not None:
        if isinstance(vars, list):
            vars = V.list_to_str(vars)
        df = df.loc[df["vars"] == vars]
    if splits is not None:
        if isinstance(splits, list):
            splits = V.list_to_str(splits)
        df = df.loc[df["splits"] == splits]
    if model is not None:
        df = df.loc[df["model"] == model]
    return df.reset_index(drop=True)

def find_trial(vars: Typ.Union[str, list[V]], splits: Typ.Union[str, list[V]], model: str) -> T:
    df = get_trials(vars, splits, model)
    if len(df) > 1:
        raise ValueError("Arguments don't specify a unique trial.")
    elif len(df) == 0:
        raise ValueError("No trial matches the arguments.")
    return df.iloc[0].loc["trial"]

def find_res_max(vars: Typ.Union[str, list[V]], splits: Typ.Union[str, list[V]], models: Typ.Optional[str]=None) -> int:
    """ Returns the highest residual for the vars"""
    res_max = -1
    for m in models:
        print((vars, splits, m))
        trial = find_trial(vars, splits, m)
        for s in trial.slices.slices:
            preds = trial.get_predictions(s)
            reals = s.df.loc[:,"sp-BLEU"].to_numpy()
            res = (preds - reals) ** 2
            res_max = max(np.max(res), res_max)
    
    return res_max

init_trials()