import warnings

import experiment.assess as A
import experiment.info as I
# import experiment.analyze as A
import experiment.run as R
# import experiment.setup as S
from slicing.variable import Variable as V
import util as U

warnings.filterwarnings("error")
U.clear_sheet(U.SHEETS["assessment"])
R.run_on_experiments(A.assess_trials, vars_list=[[V.TRAIN_SIZE], [V.TRAIN_JSD]])
