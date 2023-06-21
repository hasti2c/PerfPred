from expr import all
from kfold.kfold import *

for expr in all.C_trials:
  print([var.title for var in expr.slices.vary], expr.model.f.__name__)
  expr.fit_all()
  # expr.read_all_fits()
  expr.plot_all()
  expr.analyze_all()