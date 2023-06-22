from trial import *
from expr.language import *
from kfold.kfold import *
from compare import *

sheet_trials = [
  all.all_trials,
  all.A_trials,
  all.B_trials,
  all.C_trials,
  all.size1,
  all.size2,
  all.sizes,
  all.jsd1,
  all.jsd2,
  all.jsds,
  all.lang_1var,
  all.lang_2var,
  all.lang_3var,
  all.lang_many_var,
  all.linear_trials
]

for i, trials in enumerate(sheet_trials):
  cmpr = Comparer(trials)
  cmpr.compare_costs(i)

# for expr in all.C_trials:
#   print([var.title for var in expr.slices.vary], expr.model.f.__name__)
#   expr.fit_all()
#   # expr.read_all_fits()
#   expr.plot_all()
#   # expr.analyze_all()

# expr = MultiLanguageTrial([Var.INV_DIST, Var.PHO_DIST, Var.SYN_DIST], Model.linear(3), trial="trial1")
# expr = all.jsd1[0]
# expr.fit_all()
# expr.plot_all()

# expr = all.jsds[0]
# expr.fit_all()
# expr.plot_all()