import pandas as pd

import experiment.run as R
import experiment.setup as S
import util as U

import warnings
warnings.filterwarnings("error")

U.clear_all_sheets()
R.run_analysis(suppress=True)

# plot_vars = ["size", "nsize", "jsd"]
# for vars in plot_vars:
#   trials = S.get_trials(vars)
#   R.run_on_experiments(R.A.plot_compact, trials=trials)
# print("hi")