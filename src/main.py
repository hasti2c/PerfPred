import warnings

import experiment.compare as C
import experiment.plot as P
import experiment.run as R
import experiment.setup as S
import util as U
from modeling.trial import Trial as Tr

warnings.filterwarnings("error")
R.run_on_trials(Tr.read_or_fit)
U.clear_sheet(U.COSTS_SHEET)
C.generalized_comparison()
# C.detailed_comparison()
# C.generalized_comparison()
# R.run_on_trials(Tr.plot)
# R.run_on_experiments(P.plot_compact)