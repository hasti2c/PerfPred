import warnings

import run
from trial import Trial as Tr

warnings.filterwarnings("error")
run.init_all()
run.run_on_all(Tr.fit_all)
# run.run_on_all(lambda t: Tr.grid_search(t, (-1, 1), 5, ("loo", "mean")), models=["gm"], suppress=False)
# run.run_on_all(Tr.fit_all, models=["poly2", "poly3", "gm"]) 
# run.compare_all_costs()