import pandas as pd
import os
import sys

from split import Var as V
from model import Model as M
from trial import Trial as T
import expr.func as F

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
    row["expr"] = T(vars, model, path, trial)
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
        sys.stderr.flush ()

init_all()
run_on_all(T.fit_all, trials=["log", "power", "mult", "hybrid_mult"])
run_on_all(T.plot_all, trials=["log", "power", "mult", "hybrid_mult"])