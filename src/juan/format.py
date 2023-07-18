import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np

import experiment.run as R
import experiment.setup as S
import util as U
from modeling.trial import Trial as Tr
from slicing.variable import Variable as V

R.run_on_all(Tr.read_all_fits)

def get_predictions(expr, splits, vars):
  trials = S.get_trials([expr], [splits], [vars]).reset_index()
  slices = trials.loc[0, "trial"].slices
  
  dfs = {}
  for i, slice in enumerate(slices.slices):
    dfs[slice.__repr__()] = slice.df[[var.title for var in slice.vary] + ["sp-BLEU"]].copy().reset_index()
    for model in S.MODELS:
      trial = trials[trials["model"] == model].reset_index().loc[0, "trial"]
      fit = trial.df.loc[i, trial.model.pars].to_numpy(dtype=float)
      x = slice.x(trial.xvars)
      dfs[slice.__repr__()][model] = trial.model.f(fit, x)
  print(dfs)

PREDICTIONS = get_predictions("2A", "test", "size")