import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import experiment.run as R
import experiment.setup as S
from modeling.trial import Trial as Tr

R.run_on_trials(Tr.read_fits)

def get_predictions(splits, vars):
  trials = S.get_trials([vars], [splits]).reset_index()
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