import math
import os
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

import experiment.setup as S
from slicing.variable import Variable as V

MODEL_NAMES = {
  "linear": "Linear",
  "poly2": "Polynomial of Degree 2",
  "poly3": "Polynomial of Degree 3",
  "exp": "Exponential",
  "log": "Logarithmic",
  "power": "Power",
  "scaling": "Scaling Law"
}

VARIABLE_NAMES = {
  V.TRAIN: "Train Set",
  V.TRAIN_SIZE: "Train Set Size (thousands of tokens)",
  V.TRAIN_NORM_SIZE: "Normalized Train Set Size",
  V.TEST: "Test Set",
  V.TRAIN_JSD: "JSD of Train and Test Set",
  V.LANG: "Target Language"
}

VALUE_NAMES = {
  "gov": "Gov/PMI",
  "bible": "Bible",
  "flores": "Flores",
  "ka": "KA",
  "gu": "GU",
  "hi": "HI",
  "si": "SI",
  "ta": "TA"
}

def plot_compact(trials: pd.DataFrame, path: str=None) -> None:
  """ TODO docs """
  splits, vars = trials.iloc[0].loc["trial"].split_by, trials.iloc[0].loc["trial"].xvars
  rows = math.ceil(len(trials) / 2)
  cols = min(2, len(trials))
  fig, axes = plt.subplots(rows, cols)
  fig.tight_layout()
  fig.set_size_inches((4 * cols, 3 * rows))
  for i, j in product(range(rows), range(cols)):
    k = 2 * i + j
    ax = axes[i][j] if rows > 1 else axes[j] if cols > 1 else axes
    if k < len(trials):
      trial = trials.iloc[k].loc["trial"]
      trial.plot_together(ax, title=False, legend=False, xlabel=lambda v: VARIABLE_NAMES[v], 
                          legend_labels=lambda x: VALUE_NAMES[x])
      ax.set_title(MODEL_NAMES[trials.iloc[k].loc["model"]])
      ax.set_ylim((-10, 60))
    else:
      fig.delaxes(ax)

  ax = axes[0][0] if rows > 1 else axes[0] if cols > 1 else axes
  handles, labels = ax.get_legend_handles_labels()
  lgnd = fig.legend(handles, labels, loc='center right', 
                    title=V.list_to_str(V.complement(trials.iloc[0].loc["trial"].slices.vary), lambda v: VARIABLE_NAMES[v]))
  fig.canvas.draw()
  lgnd_dims = lgnd.get_window_extent().height, lgnd.get_window_extent().width
  fig_dims = fig.get_window_extent().height, fig.get_window_extent().width
  lgnd.remove()
  fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1 + lgnd_dims[1] / fig_dims[1], 0.5),
             title=V.list_to_str(V.complement(trials.iloc[0].loc["trial"].slices.vary), lambda v: VARIABLE_NAMES[v]))
  fig.tight_layout()
  if path is None:
      path = os.path.join(S.get_path(vars, splits), "plots.png")
  fig.savefig(path, bbox_inches = "tight")
  plt.close(fig)

def plot_individual(trials: pd.DataFrame, path: str=None) -> None:
  splits, vars = trials.iloc[0].loc["trial"].split_by, trials.iloc[0].loc["trial"].xvars
  models = trials["model"].unique()
  for model in models:
    trial = trials[trials["model"] == model]
    plot_compact(pd.DataFrame(trial), os.path.join(S.get_path(vars, splits, model), V.list_to_str(vars) + ".png"))
