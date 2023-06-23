import os
import sys

import numpy as np
import pandas as pd

import modeling.func as F
from modeling.model import Model as M
from slicing.split import Variable as V
from trial import Trial as Tr
import slicing.util as U

TRIALS = pd.DataFrame(columns=["type", "vars", "trial", "expr"])

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

def init_trial(expr, vars, trial):
    row = {"type": expr, "vars": vars, "trial": trial}
    model = MODELS[trial](len(vars))
    path = os.path.join("results", expr, "+".join([var.short for var in vars]), trial)
    row["expr"] = Tr(vars, model, path, trial)
    TRIALS.loc[len(TRIALS.index)] = row    

def init_all():
    for expr in TRIAL_TYPES:
        for vars in TRIAL_TYPES[expr]:
            for trial in MODELS:
                init_trial(expr, vars, trial)

def run_on_all(f, types=None, trials=None):
    df = TRIALS
    if types:
        df = df.loc[df["type"].isin(types)]
    if trials:
        df = df.loc[df["trial"].isin(trials)]
    for expr in df["expr"]:
        try:
            f(expr)
        except Exception as e:
            print(f"{f.__name__} on {expr} gives error: {e}.", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()

def get_stats_df():
    df = TRIALS[["type", "vars", "trial"]].copy()
    df["vars"] = ["+".join(map(V.__repr__, vars)) for vars in df["vars"]]
    df["# of slices"] = [trial.slices.N for trial in TRIALS["expr"]]
    df["avg slice size"] = [np.mean([len(slice.df) for slice in trial.slices.slices]) for trial in TRIALS["expr"]]
    df["mean"] = [trial.df["cost"].mean() for trial in TRIALS["expr"]]
    df["min"] = [trial.df["cost"].min() for trial in TRIALS["expr"]]
    df["Q1"] = [trial.df["cost"].quantile(0.25) for trial in TRIALS["expr"]]
    df["median"] = [trial.df["cost"].quantile(0.5) for trial in TRIALS["expr"]]
    df["Q3"] = [trial.df["cost"].quantile(0.75) for trial in TRIALS["expr"]]
    df["max"] = [trial.df["cost"].max() for trial in TRIALS["expr"]]
    df["var"] = [trial.df["cost"].var() for trial in TRIALS["expr"]]
    df["SD"] = [trial.df["cost"].std() for trial in TRIALS["expr"]]
    return df.round(decimals=4)

def compare_costs():
    df = get_stats_df()
    U.write_to_sheet(df, "Experiment 1 Results", 0)

init_all()
# run_on_all(Tr.fit_all, trials=["log", "power", "mult", "hybrid_mult"])
# run_on_all(Tr.plot_all, trials=["log", "power", "mult", "hybrid_mult"])
run_on_all(Tr.read_all_fits)
compare_costs()