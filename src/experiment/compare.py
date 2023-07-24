import os
import sys
import time
from itertools import product

import numpy as np
import pandas as pd

import evaluation.choose as C
import experiment.setup as S
from slicing.variable import Variable as V
import util as U


def get_costs_df():
    df = S.TRIALS[["expr", "splits", "vars", "model"]].copy()
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
        df[costs[cost] + " " + col] = [cols[col](trial.df[cost]) for trial in S.TRIALS["trial"]]
    
    df["# of slices"] = [trial.slices.N for trial in S.TRIALS["trial"]]
    slice_cols = {"mean": np.mean, "min": np.min, "max": np.max}
    for col in slice_cols:
        df[f"{col} slice size"] = [slice_cols[col]([len(slice.df) for slice in trial.slices.slices]) for trial in S.TRIALS["trial"]]
    return df.round(decimals=4)


def all_trial_costs(df, col):
    stats = [trial.df[col].set_axis(trial.slices.repr_ids()).rename(i) for i, trial in df["trial"].items()]
    return df.merge(pd.DataFrame(stats), left_index=True, right_index=True).set_index("model").drop(columns=["expr", "splits", "vars", "trial"])

def describe_trial_costs(df, col):
    stats = []
    for i, trial in df["trial"].items():
        try:
            stats.append(trial.df[col].describe().rename(i))
        except AttributeError:
            continue
    return df.merge(pd.DataFrame(stats), left_index=True, right_index=True).drop(columns=["count", "std"])

def describe_results(df, name):
    stats = df.describe().drop(index=["count", "std"])
    return pd.Series(np.diag(stats), index=stats.columns).rename(name)

def compare_models(df):
    return pd.DataFrame([describe_results(df[df["model"] == model], model) for model in S.MODELS])

def partition_all(df):
    secs = {"all": slice(None)}
    for expr in S.SPLITS:
        secs[expr] = df["expr"].isin([expr + subexpr for subexpr in S.VARS])
    for subexpr in S.VARS:
        secs[subexpr] = df["expr"].isin([expr + subexpr for expr in S.SPLITS])
    for expr, subexpr in product(S.SPLITS, S.VARS):
        expr_name = expr + subexpr
        secs[os.path.join(expr_name, expr_name)] = df["expr"] == expr_name
    secs.update(partition_by_vars(df))
    return secs

def partition_by_vars(df):
    secs = {}
    for expr, subexpr in product(S.SPLITS, S.VARS):
        expr_name = expr + subexpr
        for splits, vars in product(S.SPLITS[expr], S.VARS[subexpr]):
            split_names, var_names = V.get_var_list_name(splits), V.get_var_list_name(vars)
            secs[os.path.join(expr_name, split_names, var_names)] = (df["splits"] == split_names) & (df["vars"] == var_names)
    return secs

def save_section(df, name, path, index):
    df.to_csv(os.path.join(path, name + ".csv"))
    if U.WRITE_TO_SHEET:
        try:
            U.write_to_sheet(df, U.COSTS_SHEET, index, name)
        except U.gspread.exceptions.APIError:
            print("Sleeping for 60 seconds...", file=sys.stderr)
            time.sleep(60)
            U.write_to_sheet(df, U.COSTS_SHEET, index, name)

def choose_for_section(df):
    pareto = C.pareto(df.values)
    pareto_df = df.iloc[pareto]
    rawlsian = C.rawlsian(pareto_df.values) if len(pareto) else []
    return df.index[pareto], pareto_df.index[rawlsian]

def detailed_comparison():
    path = os.path.join(U.DATA_PATH, "analysis", "kfold rmse", "detailed")
    secs = partition_by_vars(S.TRIALS)
    for i, name in enumerate(secs):
        sec_df = S.TRIALS[secs[name]].copy()
        if sec_df.empty:
            continue
        cost_df = all_trial_costs(sec_df, "kfold rmse")
        pareto, rawlsian = choose_for_section(cost_df)
        cost_df.insert(0, "pareto", [i in pareto for i in cost_df.index])
        cost_df.insert(1, "rawlsian", [i in rawlsian for i in cost_df.index])
        save_section(cost_df, name, path, i)

def generalized_results():
    path = os.path.join(U.DATA_PATH, "analysis", "kfold rmse", "generalized")
    df = describe_trial_costs(S.TRIALS, "kfold rmse")
    df.to_csv(os.path.join(path, "results.csv"))
    if U.WRITE_TO_SHEET:
        U.write_to_sheet(pd.concat([df, df.describe()]), U.RESULTS_SHEET, U.RESULTS_PAGE, index=False)
    return df

def generalized_comparison():
    path = os.path.join(U.DATA_PATH, "analysis", "kfold rmse", "generalized")
    df = generalized_results()
    secs = partition_all(df)
    for i, name in enumerate(secs):
        sec_df = df[secs[name]].copy()
        if sec_df.empty:
            continue
        cmp_df = compare_models(sec_df)
        pareto, rawlsian = choose_for_section(cmp_df)
        cmp_df.insert(0, "pareto", [i in pareto for i in cmp_df.index])
        cmp_df.insert(1, "rawlsian", [i in rawlsian for i in cmp_df.index])
        save_section(cmp_df, name, path, i)
    return df