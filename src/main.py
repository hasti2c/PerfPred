from expr import all
from trial import *
from expr.language import *
from kfold.kfold import *

for expr in all.C_trials:
  print([var.title for var in expr.slices.vary], expr.model.f.__name__)
  expr.fit_all()
  # expr.read_all_fits()
  expr.plot_all()
  # expr.analyze_all()

# expr = MultiLanguageTrial([Var.INV_DIST, Var.PHO_DIST, Var.SYN_DIST], Model.linear(3), trial="trial1")
# expr = all.jsd1[0]
# expr.fit_all()
# expr.plot_all()

# expr = all.jsds[0]
# expr.fit_all()
# expr.plot_all()