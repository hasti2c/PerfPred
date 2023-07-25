import math
import os

import matplotlib.pyplot as plt

import experiment.setup as S
import util as U
from slicing.variable import Variable as V


def plot_compact(expr, splits, vars):
  trials = S.get_trials([expr], [splits], [vars])
  fig, axes = plt.subplots(math.ceil(len(trials) / 3), 3)
  fig.tight_layout()
  fig.set_size_inches((12, 9))
  for i, row in trials.reset_index().iterrows():
    trial = row["trial"]
    trial.plot_all_together(axes[math.floor(i / 3)][i % 3], legend=False)
  handles, labels = axes[0][0].get_legend_handles_labels()
  # plt.ylim([0, 5])
  fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.5, 0.5),
             title=V.get_var_list_name(V.others(trials.iloc[0].loc["trial"].slices.vary)))
  fig.savefig(os.path.join(U.DATA_PATH, "results", expr, splits, vars, "plots.png"))