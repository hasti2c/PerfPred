from expr.trial_util import *
from expr.train_size import *
from expr.domain_div import *

# expr = SingleSize(1, linear_single, np.zeros(2), pars=["alpha", "C"],
#                    trial="trial1", verbose=2)
# # fits, costs = expr.fit_all() # Fit.
# fits, costs = expr.read_all_fits()
# # expr.plot_all() # Plot "slice plots".
# expr.analyze_all() # Plot "analysis plots".

# expr = DoubleSize(linear_double, np.zeros(3), pars=["beta1", "beta2", "C"],
#                    trial="trial1", verbose=2)
# fits, costs = expr.fit_all()
# # fits, costs = expr.read_all_fits()
# expr.plot_all()
# expr.analyze_all()

expr = DoubleDomain(linear_double, np.zeros(3), pars=["beta1", "beta2", "C"],
                    trial="trial1", verbose=2)
expr.fit_all()
# expr.read_all_fits()
expr.plot_all()
expr.analyze_all()