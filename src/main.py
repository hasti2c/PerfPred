import pandas as pd

import experiment.run as R
import experiment.setup as S
import util as U

import warnings
warnings.filterwarnings("error")

# trial = S.find_trial([S.V.TRAIN_JSD], [S.V.LANG], "exp")
# trial.plot_together()

plot_vars = ["size", "nsize", "jsd"]
for vars in plot_vars:
  trials = S.get_trials(vars)
  # R.run_on_trials(R.T.plot_together, trials=trials)
  # R.run_on_experiments(R.P.plot_individual, trials=trials)
# # print("hi")

# trials = S.get_trials("size", "lang")
# # R.run_on_trials(R.T.plot_together, trials=trials)
# R.run_on_experiments(R.P.plot_compact, trials=trials)
# trials = S.get_trials("jsd", "lang")
# # R.run_on_trials(R.T.plot_together, trials=trials)
# R.run_on_experiments(R.P.plot_compact, trials=trials)

# sg = S.SG.get_instance(S.V.complement([S.V.LANG]))
# R.P.plot_scatter(sg, [S.V.TRAIN_SIZE], S.V.TRAIN_SIZE)

for vars in plot_vars:
  trials = S.get_trials(vars=vars, model="linear")
  for trial in trials["trial"]:
    for horiz in trial.xvars:
      if horiz.short in plot_vars:
        print(trial.slices, horiz)
        R.P.plot_scatter(trial.slices, trial.xvars, horiz)