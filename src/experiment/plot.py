import math
import os

import matplotlib.pyplot as plt

import experiment.setup as S
import util as U


def plot_compact(expr, splits, vars):
  trials = S.get_trials([expr], [splits], [vars])
  fig, axes = plt.subplots(3, math.ceil(len(trials) / 3))
  fig.tight_layout()
  fig.set_size_inches((12, 9))
  for i, row in trials.reset_index().iterrows():
    trial = row["trial"]
    trial.plot_all_together(axes[math.floor(i / 3)][i % 3])
  fig.savefig(os.path.join(U.DATA_PATH, "results", expr, splits, vars, "plots.png"))