import warnings

import run
from trial import Trial as Tr
import util as U

warnings.filterwarnings("error")
run.init_all()
run.run_on_all(Tr.read_all_fits)
# U.clear_sheet(U.COSTS_SHEET)
# run.run_on_all(Tr.plot_all)
# compare.generalized_results(True)
# compare.run_detailed_comparison()
compare.run_generalized_comparison()
