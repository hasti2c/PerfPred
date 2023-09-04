import math
import os
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

import experiment.choose as C
import experiment.setup as S
import util as U
from slicing.variable import Variable as V

def get_all_costs(trials: pd.DataFrame, col: str) -> pd.DataFrame:
    """ Returns a dataframe containing the value of column col of the trials per slice.
    Pre-condition: The trials in trials differ only by model.
    """
    stats = [trial.df[col].set_axis(trial.slices.repr_ids()).rename(i) for i, trial in trials["trial"].items()]
    return trials.merge(pd.DataFrame(stats), left_index=True, right_index=True).set_index("model").drop(columns=["vars", "splits", "trial"])

def compare_costs(trials: pd.DataFrame) -> None:
    """ Creates a table of costs of the trials per slice, with columns indicating pareto and/or rawlsian choices.
    Writes this table to a csv file. If U.WRITE_TO_SHEET is true, writes to the "costs" google sheet.
    Pre-condition: The trials in trials differ only by model.
    """
    vars, splits = trials.iloc[0].loc["trial"].xvars, trials.iloc[0].loc["trial"].split_by
    costs = get_all_costs(trials, "kfold rmse")
    pareto, rawlsian = C.choose(costs)
    costs.insert(0, "pareto", [i in pareto for i in costs.index])
    costs.insert(1, "rawlsian", [i in rawlsian for i in costs.index])
    path = os.path.join(S.get_path(vars, splits), "costs.csv")
    costs.to_csv(path)
    if U.WRITE_TO_SHEET:
        U.write_to_sheet(costs, U.SHEETS["costs"], os.path.join(V.list_to_str(vars), V.list_to_str(splits)))

def get_cost_stats(trials: pd.DataFrame, col: str=None) -> pd.DataFrame:
    """ Returns a dataframe containing statistics (mean, min, Q1, median, Q3, max) of column col of the trials (across slices).
    Pre-condition: The trials in trials differ only by model.
    """
    df = get_all_costs(trials, col)
    return df.apply(pd.Series.describe, axis=1).drop(columns=["count", "std"])

def compare_cost_stats(trials: pd.DataFrame) -> None:
    """ Creates a table of cost statistics (mean, min, Q1, median, Q3, max) of the trials (across slices).
    Writes this table to a csv file. If U.WRITE_TO_SHEET is true, writes to the "cost stats" google sheet.
    Pre-condition: The trials in trials differ only by model.
    """
    vars, splits = trials.iloc[0].loc["trial"].xvars, trials.iloc[0].loc["trial"].split_by
    stats = get_cost_stats(trials, "kfold rmse")
    path = os.path.join(S.get_path(vars, splits), "cost_stats.csv")
    stats.to_csv(path)
    if U.WRITE_TO_SHEET:
        U.write_to_sheet(stats, U.SHEETS["cost stats"], os.path.join(V.list_to_str(vars), V.list_to_str(splits)))

def get_mean_costs(trials: pd.DataFrame, col: str) -> pd.DataFrame:
    stats = [trial.df[col].mean() for trial in trials["trial"]]
    return pd.concat([trials, pd.Series(stats, name="mean kfold rmse")], axis=1)

def create_cost_table(trials: pd.DataFrame) -> None:
    vars = trials.iloc[0].loc["trial"].xvars
    all_splits, all_models = list(trials["splits"].unique()), list(trials["model"].unique())
    df = get_mean_costs(trials, "kfold rmse").drop(columns=["vars", "trial"])
    df = pd.pivot_table(df, values=["mean kfold rmse"], index=["model"], columns=["splits"])
    df.columns = df.columns.droplevel()
    df = df.reindex(columns=all_splits, index=all_models)
    path = os.path.join(S.get_path(vars), "cost_table.csv")
    df.to_csv(path)
    if U.WRITE_TO_SHEET:
        U.write_to_sheet(df, U.SHEETS["cost table"], V.list_to_str(vars))
    
def plot_compact(trials: pd.DataFrame, path: str=None) -> None:
  """ TODO docs """
  splits, vars = trials.iloc[0].loc["trial"].split_by, trials.iloc[0].loc["trial"].xvars
  rows = math.ceil(len(trials) / 3)
  cols = min(3, len(trials))
  fig, axes = plt.subplots(rows, cols)
  fig.tight_layout()
  fig.set_size_inches((4 * cols, 3 * rows))
  for i, j in product(range(rows), range(cols)):
    k = 3 * i + j
    ax = axes[i][j] if rows > 1 else axes[j]
    if k < len(trials):
      trial = trials.iloc[k].loc["trial"]
      trial.plot_together(ax, title=False, legend=False)
      ax.set_title(trials.iloc[k].loc["model"])
      ax.set_ylim((-10, 60))
    else:
      fig.delaxes(ax)

  ax = axes[0][0] if rows > 1 else axes[0]
  handles, labels = ax.get_legend_handles_labels()
  lgnd = fig.legend(handles, labels, loc='center right', 
                    title=V.list_to_str(V.complement(trials.iloc[0].loc["trial"].slices.vary)))
  fig.canvas.draw()
  lgnd_dims = lgnd.get_window_extent().height, lgnd.get_window_extent().width
  fig_dims = fig.get_window_extent().height, fig.get_window_extent().width
  lgnd.remove()
  fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1 + lgnd_dims[1] / fig_dims[1], 0.5),
             title=V.list_to_str(V.complement(trials.iloc[0].loc["trial"].slices.vary)))
  fig.tight_layout()
  if path is None:
      path = os.path.join(S.get_path(vars, splits), "plots.png")
  fig.savefig(path, bbox_inches = "tight")
  plt.close(fig)