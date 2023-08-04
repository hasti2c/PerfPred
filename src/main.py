import warnings
import pandas as pd

import experiment.analyze as A
import experiment.run as R
import experiment.setup as S
import util as U

warnings.filterwarnings("error")
U.clear_sheet(U.SHEETS["costs"])
U.clear_sheet(U.SHEETS["cost stats"])
R.run_on_experiments(A.compare_costs)
R.run_on_experiments(A.compare_cost_stats)