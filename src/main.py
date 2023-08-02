import warnings

import experiment.compare as C
import experiment.plot as P
import experiment.run as R
import experiment.setup as S
import util as U
from modeling.trial import Trial as Tr

warnings.filterwarnings("error")
R.run_on_trials(Tr.read_or_fit)
C.detailed_comparison()