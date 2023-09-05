import os

import numpy as np
import pandas as pd
import scipy.stats as sp
import sklearn.metrics as skl

import assessment.fitassesment as fa
import experiment.setup as S
import util as U
from slicing.variable import Variable as V


def pearson(true_y, pred_y):
    if len(true_y) < 8:
        return pd.NA
    return sp.normaltest(true_y - pred_y).pvalue

def shapiro(true_y, pred_y):
    return sp.shapiro(true_y - pred_y).pvalue
    
def log_likelihood(true_y, pred_y):
    sigma = np.sqrt(np.mean(abs(true_y - pred_y) ** 2))
    return np.sum(np.log(sp.norm.pdf(true_y - pred_y, loc=0, scale=sigma)))

def aic(llh):
    return 2 * (1 - llh)

def bic(llh, n):
    return 2 * (np.log(n) - llh)

def r2(true_y, pred_y):
    return skl.r2_score(true_y, pred_y)

def levene(true_y, pred_y, n_splits=2):
    if len(true_y) < 5:
        return pd.NA
    split = np.array_split(np.sort(true_y - pred_y), n_splits)
    return sp.levene(*split).pvalue

def bartlett(true_y, pred_y, n_splits=2):
    if len(true_y) < 5:
        return pd.NA
    split = np.array_split(np.sort(true_y - pred_y), n_splits)
    return sp.bartlett(*split).pvalue

def assess_trials(trials: pd.DataFrame):
    vars, splits = trials.iloc[0].loc["trial"].xvars, trials.iloc[0].loc["trial"].split_by
    xvars, slices = trials.iloc[0].loc["trial"].xvars, trials.iloc[0].loc["trial"].slices.slices
    for i, slice in enumerate(slices):
        df = pd.DataFrame(columns=["model", "log likelihood", "R2 coefficient", "normality pearson p-value", 
                                   "normality shapiro p-value", "homocedasticity levene p-value", 
                                   "homocedasticity bartlett p-value"])
        x, true_y = slice.x(xvars), slice.y
        for _, trial_row in trials.iterrows():
            trial = trial_row["trial"]
            fit = trial.df.loc[i, trial.model.pars].to_numpy(dtype=float)
            pred_y = trial.model.f(fit, x)
            
            row = {"model": trial_row["model"]}
            row["log likelihood"] = log_likelihood(true_y, pred_y)
            row["R2 coefficient"] = r2(true_y, pred_y)
            row["normality pearson p-value"] = pearson(true_y, pred_y)
            row["normality shapiro p-value"] = shapiro(true_y, pred_y)
            row["homocedasticity levene p-value"] = levene(true_y, pred_y)
            row["homocedasticity bartlett p-value"] = bartlett(true_y, pred_y)
            df.loc[len(df.index)] = row
        df.set_index("model", inplace=True)
        path = os.path.join(S.get_path(vars, splits), "assessment")
        if not os.path.exists(path):
            os.makedirs(path)
        df.to_csv(os.path.join(path, f"{slice}.csv"))
        if U.WRITE_TO_SHEET:
            U.write_to_sheet(df, U.SHEETS["assessment"], os.path.join(V.list_to_str(vars), V.list_to_str(splits), str(slice)))