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
  R.run_on_trials(R.T.plot_together, trials=trials)
  # R.run_on_experiments(R.A.plot_compact, trials=trials)
# print("hi")