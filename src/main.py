import warnings

import experiment.analyze as A
import experiment.run as R
import util as U

warnings.filterwarnings("error")
U.clear_sheet(U.SHEETS["baselines"])
R.run_on_experiments(A.compare_to_baselines)