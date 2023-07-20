import warnings

import util as U
import evaluation.compare as C
import experiment.run as R
from modeling.trial import Trial as Tr

warnings.filterwarnings("error")
U.clear_sheet(U.COSTS_SHEET)
R.run_on_all(Tr.read_all_fits)
# C.generalized_results()
# C.detailed_comparison()
C.generalized_comparison()