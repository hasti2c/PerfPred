import warnings

import experiment.plot as P
import experiment.run as R
import experiment.setup as S
import experiment.compare as C
from modeling.trial import Trial as Tr

warnings.filterwarnings("error")
R.run_on_trials(Tr.read_or_fit)
# C.detailed_comparison()
# C.generalized_comparison()
# R.run_on_trials(Tr.plot)
# R.run_on_experiments(P.plot_compact)