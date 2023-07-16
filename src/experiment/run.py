import csv
import os
import sys
from itertools import product

import pandas as pd
from scipy.stats import pearsonr
from experiment.setup import MODELS, SPLITS, VARS

import modeling.functions as F
import util as U
from modeling.model import Model as M
from modeling.trial import Trial as Tr
from slicing.variable import Variable as V
from slicing.variable import get_var_list_name

TRIALS = pd.DataFrame(columns=["expr", "splits", "vars", "model", "trial"])

def init_trial(expr, splits, vars, model, verbose=False):
    split_names, var_names = get_var_list_name(splits), get_var_list_name(vars)
    path = os.path.join(U.DATA_PATH, "results", expr, split_names, var_names, model)
    model_obj = M.get_instance(n=len(vars), **MODELS[model])
    
    try:
        trial = Tr(vars, model_obj, splits, path, model)
    except ValueError:
        if verbose:
            print(f"Can't make a trial with xvars {var_names} and splits {split_names}.", file=sys.stderr)
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

def run_on_all(f, expr=None, splits=None, vars=None, model=None, suppress=False):
    df = TRIALS
    if expr:
        df = df.loc[df["expr"] == expr]
    if splits:
        df = df.loc[df["splits"] == splits]
    if vars:
        df = df.loc[df["vars"] == vars]
    if model:
        df = df.loc[df["model"] == model]
    for i, trial in df["trial"].items():
        try:
            f(trial)
            print(f"{f.__name__} on {TRIALS.loc[i, 'expr']}:{trial} done.")
        except Exception as e:
            print(f"{f.__name__} on {TRIALS.loc[i, 'expr']}:{trial} results in error: {e}.", file=sys.stderr)
            TRIALS.drop(index=i)
            if not suppress:
                raise e
        sys.stdout.flush()
        sys.stderr.flush()


def p_val (data, var):
    """
    data = data_na_disc.csv
    var = val to be evaluated for its correlation with sp-BLEU score
    # Instead of this, should we directly write to another gsheet of var vs p-val?
    """
    col_mapping = {
        V.TRAIN1_SIZE: 1,
        V.TRAIN1_JSD: 2,
        V.TRAIN2_SIZE: 4,
        V.TRAIN2_JSD: 5,
        V.GEO_DIST: 9,
        V.GEN_DIST: 10,
        V.SYN_DIST: 11,
        V.PHO_DIST: 12,
        V.INV_DIST: 13,
        V.FEA_DIST: 14
    }

    x = []
    y = []

    with open('data_na_disc.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            col_index = col_mapping[var]
            x.append(row[col_index])
            y.append(row[15])

    return pearsonr(x, y)