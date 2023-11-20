import pandas as pd

import experiment.run as R
import experiment.setup as S
from modeling.trial import Trial as T
import experiment.assess as A
import util as U

# read_or_fit is automatically called when importing experiment.run:
#   If already fitted, fits are read from file.
#   If new, models are fitted.
# To force fitting even if already fitted, uncomment the next line.
# R.run_on_trials(T.fit) 

# assessment
for vars, splits in zip(S.VARS_LIST, S.SPLITS_LIST):
  R.run_on_experiments(A.assess_trials)

plot_vars = ["size", "nsize", "jsd"] # Only plotting for these vars since they are move relevant.
for vars in plot_vars:
  trials = S.get_trials(vars)
  R.run_on_trials(R.T.plot_together, trials=trials) # all models in one image
  R.run_on_experiments(R.P.plot_individual, trials=trials) # each model separately