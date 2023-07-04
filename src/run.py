import os
import sys
import csv
from configparser import ConfigParser
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

CONFIG_FILE = "config.txt"
INIT_CHOICE = ("kfold", "mean")

def read_config():
    config = ConfigParser()
    config.read(CONFIG_FILE)
    global INIT_CHOICE
    INIT_CHOICE = (config['Grid Search']['cost type'], config['Grid Search']['best choice'])

def init_trial(expr, splits, vars, model, verbose=False):
    split_names = "+".join(map(V.__repr__, splits)) if splits is not None else ""
    var_names = "+".join(map(V.__repr__, vars))

    path = os.path.join("results", expr, split_names, var_names, model)
    model_obj = M.get_instance(n=len(vars), **MODELS[model])
    
    try:
        trial = Tr(vars, model_obj, splits, path, model)
    except ValueError:
        if verbose:
            print(f"Can't make a trial with xvars {var_names} and splits {split_names}.", file=sys.stderr)
        return

    try:
        init = trial.read_grid_search(INIT_CHOICE)
        trial.model.init = np.full(trial.model.init.shape, init) # TODO make less messy
    except FileNotFoundError:
        if verbose:
            print(f"Failed reading init value for {expr}:{trial}. Using default 0.", file=sys.stderr)
    row = {"expr": expr, "splits": split_names, "vars": var_names, "model": model, "trial": trial}
    TRIALS.loc[len(TRIALS.index)] = row    

def init_all(verbose=False):
    read_config()
    for expr, subexpr in product(SPLITS, VARS):
        for splits, vars, model in product(SPLITS[expr], VARS[subexpr], MODELS):
            init_trial(expr + subexpr, splits, vars, model, verbose)

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

def get_stats_df():
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

def compare_costs(df, page, name):
    U.write_to_sheet(df, "Experiment 1 Results", page, name)

def compare_all_costs():
    stats_df = get_stats_df()
    compare_costs(stats_df, 0, "all")
    k = 1

    for expr in SPLITS:
        df = stats_df[stats_df["expr"].isin([expr + subexpr for subexpr in VARS])]
        compare_costs(df, k, expr)
        k += 1
    
    for subexpr in VARS:
        df = stats_df[stats_df["expr"].isin([expr + subexpr for expr in SPLITS])]
        compare_costs(df, k, subexpr)
        k += 1

    for expr, subexpr in product(SPLITS, VARS):
        df = stats_df[stats_df["expr"] == expr + subexpr]
        compare_costs(df, k, expr + subexpr)
        k += 1
    
    for model in MODELS:
        df = stats_df[stats_df["model"] == model]
        compare_costs(df, k, model)
        k += 1

def p_val (data, var):
    """
    data = data_na_disc.csv
    var = val to be evaluated for its correlation with sp-BLEU score
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