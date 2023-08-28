import warnings

import experiment.assess as A
import experiment.info as I
# import experiment.analyze as A
import experiment.run as R
import experiment.setup as S
import util as U

from modeling.model import Model as M
import modeling.functions as F
from modeling.trial import Trial as T
from slicing.variable import Variable as V

model = M.get_instance(F.logarithmic, 1)
trial = T([V.TRAIN_SIZE], [V.LANG], model)
fits, costs, kfs = trial.fit()
