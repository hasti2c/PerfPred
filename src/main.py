import warnings

import experiment.plot as P
import experiment.run as R
import experiment.setup as S
from modeling.trial import Trial as Tr

warnings.filterwarnings("error")
R.run_on_all(Tr.read_all_fits)
P.plot_compact("2A", "lang", "size")
# R.run_on_all(Tr.plot_all_together, exprs=["2A", "2B"])