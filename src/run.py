import os
import sys

import numpy as np
import pandas as pd

import modeling.func as F
from modeling.model import Model as M
from slicing.split import Variable as V
from trial import Trial as Tr
import slicing.util as U

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

def init_trial(expr, vars, model):
    var_names = "+".join(map(V.__repr__, vars))
    model_obj = MODELS[model](len(vars))
    path = os.path.join("results", expr, var_names, model)
    row = {"type": expr, "vars": var_names, "model": model,
           "trial": Tr(vars, model_obj, path, model)}
    TRIALS.loc[len(TRIALS.index)] = row    

def init_all():
    for expr in TRIAL_TYPES:
        for vars in TRIAL_TYPES[expr]:
            for model in MODELS:
                init_trial(expr, vars, model)

def run_on_all(f, types=None, vars=None, models=None):
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
        except Exception as e:
            print(f"{f.__name__} on {trial} gives error: {e}.", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()

def get_stats_df():
    df = TRIALS[["type", "vars", "model"]].copy()
    df["mean"] = [trial.df["loo rmse"].mean() for trial in TRIALS["trial"]]
    df["min"] = [trial.df["loo rmse"].min() for trial in TRIALS["trial"]]
    df["Q1"] = [trial.df["loo rmse"].quantile(0.25) for trial in TRIALS["trial"]]
    df["median"] = [trial.df["loo rmse"].quantile(0.5) for trial in TRIALS["trial"]]
    df["Q3"] = [trial.df["loo rmse"].quantile(0.75) for trial in TRIALS["trial"]]
    df["max"] = [trial.df["loo rmse"].max() for trial in TRIALS["trial"]]
    df["var"] = [trial.df["loo rmse"].var() for trial in TRIALS["trial"]]
    df["SD"] = [trial.df["loo rmse"].std() for trial in TRIALS["trial"]]
    df["# of slices"] = [trial.slices.N for trial in TRIALS["trial"]]
    df["avg slice size"] = [np.mean([len(slice.df) for slice in trial.slices.slices]) for trial in TRIALS["trial"]]
    df["min slice size"] = [np.min([len(slice.df) for slice in trial.slices.slices]) for trial in TRIALS["trial"]]
    df["max slice size"] = [np.max([len(slice.df) for slice in trial.slices.slices]) for trial in TRIALS["trial"]]
    return df.round(decimals=4)

def compare_costs(df, page, types=None, vars=None, models=None):
    if types:
        df = df.loc[df["type"].isin(types)]
    if vars:
        df = df.loc[df["vars"].isin(vars)]
    if models:
        df = df.loc[df["model"].isin(models)]
    U.write_to_sheet(df, "Experiment 1 Results", page)

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

init_all()
run_on_all(Tr.fit_all)
# run_on_all(Tr.plot_all)
# run_on_all(Tr.read_all_fits, types=["1B"])
# run_on_all(Tr.plot_all, types=["1B"])
compare_all_costs()
print("hi")
