import warnings

import evaluation.compare as C
import experiment.run as R
from modeling.trial import Trial as Tr

warnings.filterwarnings("error")
R.run_on_all(Tr.fit_all)
C.generalized_results()