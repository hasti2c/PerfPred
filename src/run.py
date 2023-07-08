import csv
import os
import sys
import time
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import modeling.func as F
import util as U
from modeling.model import Model as M
from slicing.split import Variable as V
from trial import Trial as Tr

TRIALS = pd.DataFrame(columns=["expr", "splits", "vars", "model", "trial"])

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
    "2": [[V.LANG], [V.TEST]]
}

MODELS = {
    "linear": {'f': F.linear},
    "poly2": {'f': F.polynomial, 'k': 2},
    "poly3": {'f': F.polynomial, 'k': 3},
    "exp": {'f': F.exponential},
    "log": {'f': F.logarithmic},
    "power": {'f': F.power},
    "mult": {'f': F.multiplicative},
    "hybrid_mult": {'f': F.hybrid_multiplicative, 'bounds': (-1000, 1000)},
    "am": {'f': F.arithmetic_mean_linear},
    "gm": {'f': F.geometric_mean_linear},
    "hm": {'f': F.harmonic_mean_linear},
}
# TODO dont use mean models for single var

def get_var_names(vars=None):
    return "+".join(map(V.__repr__, vars)) if vars is not None else ""

def init_trial(expr, splits, vars, model, verbose=False):
    split_names, var_names = get_var_names(splits), get_var_names(vars)
    path = os.path.join("results", expr, split_names, var_names, model)
    model_obj = M.get_instance(n=len(vars), **MODELS[model])
    
    try:
        trial = Tr(vars, model_obj, splits, path, model)
    except ValueError:
        if verbose:
            print(f"Can't make a trial with xvars {var_names} and splits {split_names}.", file=sys.stderr)
        return

    try:
        init = trial.read_grid_search(U.INIT_CHOICE)
        trial.model.init = np.full(trial.model.init.shape, init) # TODO make less messy
    except FileNotFoundError:
        if verbose:
            print(f"Failed reading init value for {expr}:{trial}. Using default 0.", file=sys.stderr)
    if verbose:
        print(f"Initialized {expr}:{trial}", file=sys.stderr)
    return {"expr": expr, "splits": split_names, "vars": var_names, "model": model, "trial": trial}

def init_all(verbose=False):
    for expr, subexpr in product(SPLITS, VARS):
        for splits, vars, model in product(SPLITS[expr], VARS[subexpr], MODELS):
            row = init_trial(expr + subexpr, splits, vars, model, verbose)
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
    for trial in df["trial"]:
        try:
            f(trial)
            print(f"{f.__name__} on {trial} done.")
        except Exception as e:
            print(f"{f.__name__} on {trial} results in error: {e}.", file=sys.stderr)
            if not suppress:
                raise e
        sys.stdout.flush()
        sys.stderr.flush()

def get_costs_df():
    df = TRIALS[["expr", "splits", "vars", "model"]].copy()
    costs = {"rmse": "simple", "kfold rmse": "KFold"}
    cols = {"mean": pd.Series.mean,
            "min": pd.Series.min,
            "Q1": lambda s: pd.Series.quantile(s, 0.25),
            "median": lambda s: pd.Series.quantile(s, 0.5),
            "Q3": lambda s: pd.Series.quantile(s, 0.75),
            "max": pd.Series.max,
            "var": pd.Series.var,
            "SD": pd.Series.std}
    for cost, col in product(costs, cols):
        df[costs[cost] + " " + col] = [cols[col](trial.df[cost]) for trial in TRIALS["trial"]]
    
    df["# of slices"] = [trial.slices.N for trial in TRIALS["trial"]]
    slice_cols = {"mean": np.mean, "min": np.min, "max": np.max}
    for col in slice_cols:
        df[f"{col} slice size"] = [slice_cols[col]([len(slice.df) for slice in trial.slices.slices]) for trial in TRIALS["trial"]]
    return df.round(decimals=4)

def describe_trials(col):
    stats = [trial.df[col].describe().rename(i) for i, trial in TRIALS["trial"].items()]
    return TRIALS.merge(pd.DataFrame(stats), left_index=True, right_index=True).drop(columns="count")

def describe_results(df, name):
    stats = df.describe().drop("count")
    return pd.Series(np.diag(stats), index=stats.columns).rename(name)

def compare_models(df):
    return pd.DataFrame([describe_results(df[df["model"] == model], model) for model in MODELS])

def run_comparison():
    path = os.path.join("analysis", "kfold rmse")
    df = describe_trials("kfold rmse")
    df.to_csv(os.path.join(path, "results.csv"))
    secs = {"all": slice(None)}
    for expr in SPLITS:
        secs[expr] = df["expr"].isin([expr + subexpr for subexpr in VARS])
    for subexpr in VARS:
        secs[subexpr] = df["expr"].isin([expr + subexpr for expr in SPLITS])
    for expr, subexpr in product(SPLITS, VARS):
        expr_name = expr + subexpr
        secs[os.path.join(expr_name, expr_name)] = df["expr"] == expr_name
        for splits, vars in product(SPLITS[expr], VARS[subexpr]):
            split_names, var_names = get_var_names(splits), get_var_names(vars)
            secs[os.path.join(expr_name, split_names, var_names)] = (df["splits"] == split_names) & (df["vars"] == var_names)
    for i, name in enumerate(secs):
        sec_df = df[secs[name]]
        if sec_df.empty:
            continue
        cmp = compare_models(sec_df)
        cmp.to_csv(os.path.join(path, name + ".csv"))
        if U.WRITE_TO_SHEET:
            try:
                U.write_to_sheet(cmp, U.COSTS_SHEET, i, name)
            except U.gspread.exceptions.APIError:
                print("Sleeping for 60 seconds...", file=sys.stderr)
                time.sleep(60)
                U.write_to_sheet(cmp, U.COSTS_SHEET, i, name)


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