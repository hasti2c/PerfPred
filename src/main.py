import warnings

import evaluation.compare as compare
import experiment.run as run
import util as U
from modeling.trial import Trial as Tr

warnings.filterwarnings("error")
run.init_all()
# run.run_on_all(lambda t: t.grid_search((-1, 1), 5, U.INIT_CHOICE))
# grid_search(self, init_range: tuple[float], num: int, init_choice: tuple[str])
run.run_on_all(Tr.fit_all)
# U.clear_sheet(U.COSTS_SHEET)
# run.run_on_all(Tr.plot_all)
compare.generalized_results(True)
# compare.run_detailed_comparison()
# compare.run_generalized_comparison()