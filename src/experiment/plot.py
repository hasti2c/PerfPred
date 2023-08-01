import math
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

import experiment.setup as S
import util as U
from slicing.variable import Variable as V


def plot_compact(trials):
  splits, vars = trials.iloc[0].loc["trial"].split_by, trials.iloc[0].loc["trial"].xvars
  fig, axes = plt.subplots(math.ceil(len(trials) / 3), 3)
  fig.tight_layout()
  fig.set_size_inches((12, 9))
  for i, j in product(range(math.ceil(len(trials) / 3)), range(3)):
    k = 3 * i + j
    if k < len(trials):
      trial = trials.iloc[k].loc["trial"]
      trial.plot_together(axes[i][j], legend=False)
      axes[i][j].set_ylim((-10, 60))
    else:
      fig.delaxes(axes[i][j])

  handles, labels = axes[0][0].get_legend_handles_labels()
  lgnd = fig.legend(handles, labels, loc='center right', 
                    title=V.list_to_str(V.others(trials.iloc[0].loc["trial"].slices.vary)))
  fig.canvas.draw()
  lgnd_dims = lgnd.get_window_extent().height, lgnd.get_window_extent().width
  fig_dims = fig.get_window_extent().height, fig.get_window_extent().width
  lgnd.remove()
  fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1 + lgnd_dims[1] / fig_dims[1], 0.5),
             title=V.list_to_str(V.others(trials.iloc[0].loc["trial"].slices.vary)))
  fig.tight_layout()
  fig.savefig(os.path.join(U.DATA_PATH, "results", "multi" if len(vars) > 1 else "", V.list_to_str(vars), V.list_to_str(splits), "plots.png"), bbox_inches = "tight")
  plt.close(fig)