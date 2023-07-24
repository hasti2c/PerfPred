import warnings

import experiment.setup as S
import experiment.run as R
import experiment.plot as P
from modeling.trial import Trial as Tr
import matplotlib.pyplot as plt

warnings.filterwarnings("error")
R.run_on_all(Tr.read_all_fits)
R.run_on_all(Tr.plot_all_together, exprs=["2A", "2B"])