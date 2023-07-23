import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
from itertools import product

import experiment.run as R
import experiment.setup as S
import util as U
from modeling.trial import Trial as Tr
import slicing.variable as V
from pprint import pprint

R.run_on_all(Tr.read_all_fits)

def get_predictions(expr, splits, vars):
  trials = S.get_trials([expr], [splits], [vars]).reset_index()
  slices = trials.loc[0, "trial"].slices
    
  dfs = {}
  for i, slice in enumerate(slices.slices):
    dfs[slice.__repr__()] = slice.df[[var.title for var in slice.vary] + ["sp-BLEU"]].copy().reset_index(drop=True)
    for model in S.MODELS:
      df = trials[trials["model"] == model]
      if len(df) == 0:
        continue
      trial = df.reset_index().loc[0, "trial"]
      fit = trial.df.loc[i, trial.model.pars].to_numpy(dtype=float)
      x = slice.x(trial.xvars)
      dfs[slice.__repr__()][model] = trial.model.f(fit, x)
  return dfs

EXPRS = [
  "1A", 
  "1B", 
  "1C", 
  "2A", 
  "2B", 
  "2C"
]
SPLITS = {}
VARS = {}
for expr, subexpr in product(S.SPLITS, S.VARS):
  SPLITS[expr + subexpr] = [V.get_var_list_name(splits) for splits in S.SPLITS[expr]]
  VARS[expr + subexpr] = [V.get_var_list_name(vars) for vars in S.VARS[subexpr]]

# print(SPLITS)
# print(VARS)
# 
# PREDS = get_predictions(EXPRS[0], SPLITS[EXPRS[0]][0], VARS[EXPRS[0]][3] )
# print(PREDS.keys())