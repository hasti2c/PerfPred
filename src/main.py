import warnings

import run
from trial import Trial as Tr
import util as U

warnings.filterwarnings("error")
run.init_all()
run.run_on_all(Tr.read_all_fits)
run.run_comparison()