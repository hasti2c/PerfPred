from trial import *
from expr.train_size import *
from expr.language import *
from kfold.kfold import *
from compare import *
from expr.func import *
from expr import all

# sheet_trials = [
#   all.all_trials,
#   all.A_trials,
#   all.B_trials,
#   all.C_trials,
#   all.size1,
#   all.size2,
#   all.sizes,
#   all.jsd1,
#   all.jsd2,
#   all.jsds,
#   all.lang_1var,
#   all.lang_2var,
#   all.lang_3var,
#   all.lang_many_var,
#   all.linear_trials
# ]

# for i, trials in enumerate(sheet_trials):
#   cmpr = Comparer(trials)
#   cmpr.compare_costs(i)

# # expr = SingleSizeTrial(1, Model.linear(1), trial="trial1")
# expr = SingleSizeTrial(1, Model.polynomial(1, 2), trial="trial2")
# # expr = SingleSizeTrial(1, Model(func.polynomial, np.zeros(3),
# #                          pars=["alpha", "beta", "C"]), trial="trial2")
# print(expr.model.pars)
# expr.fit_all()
# expr.plot_all()