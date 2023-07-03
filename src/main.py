import warnings

import run
from trial import Trial as Tr

warnings.filterwarnings("error")
run.init_all()
# run.run_on_all(Tr.fit_all, splits="test") 
# run.run_on_all(Tr.read_all_fits)
run.run_on_all(lambda t: Tr.grid_search(t, (-1, 1), 5, ("kfold", "mean")), suppress=True)
# run.compare_all_costs()