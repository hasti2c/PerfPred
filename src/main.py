import warnings

# import experiment.assess as A
import experiment.info as I
import experiment.analyze as A
import experiment.setup as S
import util as U

from slicing.variable import Variable as V
import experiment.run as R
from modeling.trial import Trial as T

import pandas as pd
pd.set_option("display.max_columns", 10)

R.run_on_experiments(A.create_cost_table, group_by_vars=True)