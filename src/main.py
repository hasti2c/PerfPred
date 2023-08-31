import warnings

import experiment.assess as A
import experiment.info as I
# import experiment.analyze as A
import experiment.setup as S
import util as U

from slicing.variable import Variable as V
import experiment.run as R
from modeling.trial import Trial as T

R.run_on_trials(lambda trial: print(trial.path))
R.run_on_trials(T.fit, splits_list=[[], [V.LANG]])