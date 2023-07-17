import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np

import run
import util as U
from slicing.split import Variable as V

splits = run.SPLITS["2"][0]
vars = run.VARS["B"][2]
langs = V.LANG.values(U.RECORDS)

trials = {}
for model in run.MODELS:
  trials[model] = run.init_trial("2B", splits, vars, model)["trial"]
  trials[model].read_all_fits()

lang_slices, lang_dfs = {}, {}
ids, slices = trials['linear'].slices.ids, trials['linear'].slices.slices
for lang in langs:
    slice = slices[ids.loc[ids["language to"] == lang].index[0]]
    lang_slices[lang] = slice
    lang_dfs[lang] = slice.df[[var.title for var in slice.vary] + ["sp-BLEU"]]

for model, trial in trials.items():
  for lang in langs:
    row = trial.df[trial.df["language to"] == lang]
    fit = np.array(row[trial.model.pars]).flatten()
    x = lang_slices[lang].x(trial.xvars)
    lang_dfs[lang][model] = trial.model.f(fit, x)

for lang in langs:
   lang_dfs[lang].reset_index(inplace=True, drop=True)
   lang_dfs[lang].to_csv(os.path.join("src", "juan", lang + ".csv"), )

PREDICTIONS = lang_dfs