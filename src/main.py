import warnings

import experiment.plot as P
import experiment.run as R
import experiment.setup as S
from modeling.trial import Trial as Tr

warnings.filterwarnings("error")
R.run_on_all(Tr.read_all_fits)
P.plot_compact("2A", "test", "size")
P.plot_compact("2A", "lang", "size")
P.plot_compact("2A", "test+lang", "size")
P.plot_compact("2B", "lang", "jsd")
P.plot_compact("2C", "test", "fea")
P.plot_compact("2C", "test", "inv")
P.plot_compact("2C", "test", "pho")
P.plot_compact("2C", "test", "syn")
P.plot_compact("2C", "test", "gen")
P.plot_compact("2C", "test", "geo")
# R.run_on_all(Tr.plot_all_together, exprs=["2A", "2B"])