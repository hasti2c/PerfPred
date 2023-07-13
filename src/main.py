import warnings

import evaluation.compare as compare
import run
import util as U
from modeling.trial import Trial as Tr

warnings.filterwarnings("error")
U.clear_sheet(U.COSTS_SHEET)
run.init_all()
run.run_on_all(Tr.read_all_fits)
compare.run_comparison()