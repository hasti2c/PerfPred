import os
import sys
from itertools import product

import numpy as np
import pandas as pd

import modeling.functions as F
import util as U
from modeling.model import Model as M
from modeling.trial import Trial as Tr
from slicing.variable import Variable as V
from slicing.variable import get_var_list_name

if U.EXPERIMENT_TYPE == "one stage":
    VARS = {
        "A": [[V.TRAIN_SIZE]],
        "B": [[V.TRAIN_JSD]],
        "C": [[V.FEA_DIST], [V.INV_DIST], [V.PHO_DIST], [V.SYN_DIST], [V.GEN_DIST], [V.GEO_DIST],
            [V.INV_DIST, V.PHO_DIST], [V.INV_DIST, V.SYN_DIST], [V.PHO_DIST, V.SYN_DIST], [V.GEN_DIST, V.GEO_DIST], 
            [V.INV_DIST, V.PHO_DIST, V.SYN_DIST], [V.FEA_DIST, V.GEN_DIST, V.GEO_DIST], 
            [V.FEA_DIST, V.INV_DIST, V.PHO_DIST, V.SYN_DIST], 
            [V.INV_DIST, V.PHO_DIST, V.SYN_DIST, V.GEN_DIST, V.GEO_DIST],
            [V.FEA_DIST, V.INV_DIST, V.PHO_DIST, V.SYN_DIST, V.GEN_DIST, V.GEO_DIST]]
    }
else:
    VARS = {
        "A": [[V.TRAIN1_SIZE], [V.TRAIN2_SIZE], [V.TRAIN1_SIZE, V.TRAIN2_SIZE]],
        "B": [[V.TRAIN1_JSD], [V.TRAIN2_JSD], [V.TRAIN1_JSD, V.TRAIN2_JSD]],
        "C": [[V.FEA_DIST], [V.INV_DIST], [V.PHO_DIST], [V.SYN_DIST], [V.GEN_DIST], [V.GEO_DIST],
            [V.INV_DIST, V.PHO_DIST], [V.INV_DIST, V.SYN_DIST], [V.PHO_DIST, V.SYN_DIST], [V.GEN_DIST, V.GEO_DIST], 
            [V.INV_DIST, V.PHO_DIST, V.SYN_DIST], [V.FEA_DIST, V.GEN_DIST, V.GEO_DIST], 
            [V.FEA_DIST, V.INV_DIST, V.PHO_DIST, V.SYN_DIST], 
            [V.INV_DIST, V.PHO_DIST, V.SYN_DIST, V.GEN_DIST, V.GEO_DIST],
            [V.FEA_DIST, V.INV_DIST, V.PHO_DIST, V.SYN_DIST, V.GEN_DIST, V.GEO_DIST]]
    }

SPLITS = {
    "1": [None],
    "2": [[V.LANG], [V.TEST]] # TODO splits=[]
}

MODELS = {
    "linear": {'f': F.linear},
    "poly2": {'f': F.polynomial, 'k': 2},
    "poly3": {'f': F.polynomial, 'k': 3},
    "exp": {'f': F.exponential, 'bounds': (-np.inf, 6)},
    "log": {'f': F.logarithmic},
    "power": {'f': F.power, 'bounds': (-250, np.inf)},
    "mult": {'f': F.multiplicative, 'bounds': (-200, np.inf)},
    "hybrid_mult": {'f': F.hybrid_multiplicative, 'bounds': (-90, 90)},
    "am": {'f': F.arithmetic_mean_linear},
    "gm": {'f': F.geometric_mean_linear},
    "hm": {'f': F.harmonic_mean_linear},
}
# TODO dont use mean models for single var

TRIALS = pd.DataFrame(columns=["expr", "splits", "vars", "model", "trial"])

def init_trial(expr, splits, vars, model, verbose=False):
    split_names, var_names = get_var_list_name(splits), get_var_list_name(vars)
    path = os.path.join(U.DATA_PATH, "results", expr, split_names, var_names, model)
    model_obj = M.get_instance(n=len(vars), **MODELS[model])
    try:
        trial = Tr(vars, model_obj, splits, path, model)
    except ValueError:
        return None
    if verbose:
        print(f"Initialized {expr}:{trial}", file=sys.stderr)
    return {"expr": expr, "splits": split_names, "vars": var_names, "model": model, "trial": trial}

def init_all(verbose=False):
    for expr, subexpr in product(SPLITS, VARS):
        for splits, vars, model in product(SPLITS[expr], VARS[subexpr], MODELS):
            row = init_trial(expr + subexpr, splits, vars, model, verbose)
            if row is not None:
                TRIALS.loc[len(TRIALS.index)] = row

def get_trials(exprs=[], splits=[], vars=[], models=[]):
    df = TRIALS
    if exprs:
        df = df.loc[df["expr"].isin(exprs)]
    if splits:
        df = df.loc[df["splits"].isin(splits)]
    if vars:
        df = df.loc[df["vars"].isin(vars)]
    if models:
        df = df.loc[df["model"].isin(models)]
    return df

init_all()