from expr.train_size import *
from expr.trial_util import *

expr = SingleTrain(1, linear_single, np.zeros(2), par_names=["alpha", "C"],
                   trial="trial1", verbose=2)
expr.fit_all()
# expr.read_all_fits()
expr.plot_all()
# expr.analyze_all()