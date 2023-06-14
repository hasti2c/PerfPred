from expr.trial_util import *
from expr.train_size import *
from expr.domain_div import *

# Single Size: D1, D2 # TODO
# Double Size: D1+D2
# Single Domain: j1, j2
# Double Domain: j1+j2

# expr = SingleTrain(1, linear_single, np.zeros(2), par_names=["alpha", "C"],
#                    trial="trial1", verbose=2)
# expr.fit_all()
# # expr.read_all_fits()
# expr.plot_all()
# # expr.analyze_all()

create_trial_folder(os.path.join(path_B, "jsd1", "trial1"))
create_trial_folder(os.path.join(path_B, "jsd2", "trial1"))
create_trial_folder(os.path.join(path_B, "jsds", "trial1"))