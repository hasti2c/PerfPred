import os
import sys
from configparser import ConfigParser
from itertools import product

import numpy as np
import pandas as pd

import modeling.func as F
import slicing.util as U
from modeling.model import Model as M
from slicing.split import Variable as V
from trial import Trial as Tr

TRIALS = pd.DataFrame(columns=["type", "vars", "model", "trial"])

TRIAL_TYPES = {
    "1A": [[V.TRAIN1_SIZE], [V.TRAIN2_SIZE], [V.TRAIN1_SIZE, V.TRAIN2_SIZE]],
    "1B": [[V.TRAIN1_JSD], [V.TRAIN2_JSD], [V.TRAIN1_JSD, V.TRAIN2_JSD]],
    "1C": [[V.FEA_DIST], [V.INV_DIST], [V.PHO_DIST], [V.SYN_DIST], [V.GEN_DIST], [V.GEO_DIST],
           [V.INV_DIST, V.PHO_DIST], [V.INV_DIST, V.SYN_DIST], [V.PHO_DIST, V.SYN_DIST], [V.GEN_DIST, V.GEO_DIST], 
           [V.INV_DIST, V.PHO_DIST, V.SYN_DIST], [V.FEA_DIST, V.GEN_DIST, V.GEO_DIST], 
           [V.FEA_DIST, V.INV_DIST, V.PHO_DIST, V.SYN_DIST], 
           [V.INV_DIST, V.PHO_DIST, V.SYN_DIST, V.GEN_DIST, V.GEO_DIST],
           [V.FEA_DIST, V.INV_DIST, V.PHO_DIST, V.SYN_DIST, V.GEN_DIST, V.GEO_DIST]]
}

MODELS = {
    "linear": M.linear,
    "poly2": lambda m: M.polynomial(m, 2),
    "poly3": lambda m: M.polynomial(m, 3),
    "exp": lambda m: M.nonlinear(m, F.exponential),
    "log": lambda m: M.nonlinear(m, F.logarithmic),
    "power": lambda m: M.nonlinear(m, F.power),
    "mult": lambda m: M.nonlinear(m, F.multiplicative),
    "hybrid_mult": lambda m: M.nonlinear(m, F.hybrid_multiplicative),
    "am": lambda _: M.mean(F.arithmetic_mean_linear),
    "gm": lambda _: M.mean(F.geometric_mean_linear),
    "hm": lambda _: M.mean(F.harmonic_mean_linear),
}
# TODO dont use mean models for single var

CONFIG_FILE = "config.txt"
INIT_CHOICE = ("loo", "mean")

def read_config():
    config = ConfigParser()
    config.read(CONFIG_FILE)
    global INIT_CHOICE
    INIT_CHOICE = (config['Grid Search']['cost type'], config['Grid Search']['best choice'])

def init_trial(expr, vars, model):
    var_names = "+".join(map(V.__repr__, vars))
    path = os.path.join("results", expr, var_names, model)
    model_obj = MODELS[model](len(vars))
    trial = Tr(vars, model_obj, path, model)
    try:
        init = trial.read_grid_search(INIT_CHOICE)
        trial.model.init = np.full(trial.model.init.shape, init) # TODO make less messy (after seeing if it works)
    except Exception as e:
        print(f"Failed reading init value for {trial}. Using default 0. Error: {e}.", file=sys.stderr)
    row = {"type": expr, "vars": var_names, "model": model, "trial": trial}
    TRIALS.loc[len(TRIALS.index)] = row    

def init_all():
    read_config()
    for expr in TRIAL_TYPES:
        for vars in TRIAL_TYPES[expr]:
            for model in MODELS:
                init_trial(expr, vars, model)

def run_on_all(f, types=None, vars=None, models=None, suppress=True):
    df = TRIALS
    if types:
        df = df.loc[df["type"].isin(types)]
    if vars:
        df = df.loc[df["vars"].isin(vars)]
    if models:
        df = df.loc[df["model"].isin(models)]
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
    df = TRIALS[["type", "vars", "model"]].copy()
    costs = {"rmse": "simple", "loo rmse": "LOO"}
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

def compare_costs(df, page, types=None, vars=None, models=None):
    if types:
        df = df.loc[df["type"].isin(types)]
    if vars:
        df = df.loc[df["vars"].isin(vars)]
    if models:
        df = df.loc[df["model"].isin(models)]
    U.write_to_sheet(df, "Experiment 1 Results", page)
    U.format_column("Experiment 1 Results", page, 4)

def compare_all_costs():
    df = get_stats_df()
    compare_costs(df, 0)
    k = 1
    for type in TRIAL_TYPES:
        compare_costs(df, k, types=[type])
        k += 1
    for model in MODELS:
        compare_costs(df, k, models=[model])
        k += 1